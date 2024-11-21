// multi_gpu_matmul_mi300x.cpp

#include <hip/hip_runtime.h>
#include <rccl.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <exception>

// Macro for checking HIP errors
#define HIP_CHECK(cmd)                                                     \
    {                                                                     \
        hipError_t error = cmd;                                           \
        if (error != hipSuccess) {                                        \
            throw std::runtime_error(std::string("HIP Error: ") +        \
                                     hipGetErrorString(error) +            \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                 \
    }

// Macro for checking RCCL errors
#define RCCL_CHECK(cmd)                                                    \
    {                                                                     \
        ncclResult_t error = cmd;                                         \
        if (error != ncclSuccess) {                                       \
            throw std::runtime_error(std::string("RCCL Error: ") +        \
                                     ncclGetErrorString(error) +           \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                 \
    }

// Function to get RCCL error strings
const char* ncclGetErrorString(ncclResult_t error) {
    switch (error) {
        case ncclSuccess:
            return "ncclSuccess";
        case ncclUnhandledCudaError:
            return "ncclUnhandledCudaError";
        case ncclSystemError:
            return "ncclSystemError";
        case ncclInternalError:
            return "ncclInternalError";
        case ncclInvalidArgument:
            return "ncclInvalidArgument";
        default:
            return "Unknown nccl error";
    }
}

// Kernel for matrix multiplication: C = A * B
// Optimized for MI300X with block size 32x32 and shared memory tiling
__global__ void matMulKernel(const float* A, const float* B, float* C, 
                             int M, int N, int K) {
    // Define tile size
    const int TILE_SIZE = 32;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate row and column indices of C element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles of A and B matrices
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load elements into shared memory with boundary checks
        if (row < M && (tile * TILE_SIZE + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if ((tile * TILE_SIZE + threadIdx.y) < N && col < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * K + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int e = 0; e < TILE_SIZE; ++e)
            value += As[threadIdx.y][e] * Bs[e][threadIdx.x];

        __syncthreads();
    }

    // Write the result to C with boundary check
    if (row < M && col < K)
        C[row * K + col] = value;
}

// Utility function to perform matrix multiplication on the host for validation
void matMulCPU(const std::vector<float>& A, const std::vector<float>& B, 
              std::vector<float>& C, int M, int N, int K) {
    for(int row = 0; row < M; ++row){
        for(int col = 0; col < K; ++col){
            float value = 0.0f;
            for(int e = 0; e < N; ++e){
                value += A[row * N + e] * B[e * K + col];
            }
            C[row * K + col] = value;
        }
    }
}

// Function to initialize matrices with random data
void initializeMatrix(std::vector<float>& mat) {
    for(auto &val : mat){
        val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// RAII Wrapper for HIP Streams
class HIPStream {
public:
    HIPStream() {
        HIP_CHECK(hipStreamCreate(&stream_));
    }

    ~HIPStream() {
        if(stream_){
            hipError_t err = hipStreamDestroy(stream_);
            if(err != hipSuccess){
                std::cerr << "Failed to destroy HIP stream: " << hipGetErrorString(err) << "\n";
            }
            stream_ = nullptr;
        }
    }

    hipStream_t get() const { return stream_; }

private:
    hipStream_t stream_ = nullptr;
};

// RAII Wrapper for Device Memory
class DeviceMemory {
public:
    DeviceMemory(size_t size) : size_(size) {
        HIP_CHECK(hipMalloc(&ptr_, size_));
    }

    ~DeviceMemory() {
        if(ptr_){
            hipError_t err = hipFree(ptr_);
            if(err != hipSuccess){
                std::cerr << "Failed to free device memory: " << hipGetErrorString(err) << "\n";
            }
            ptr_ = nullptr;
        }
    }

    float* get() const { return ptr_; }

    // Disable copy semantics
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Enable move semantics
    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if(this != &other){
            // Free existing memory
            if(ptr_){
                hipFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    float* ptr_ = nullptr;
    size_t size_ = 0;
};

// RAII Wrapper for RCCL Communicators
class RCCLCommunicator {
public:
    RCCLCommunicator(int num_gpus) : num_gpus_(num_gpus), comms_(num_gpus, nullptr) {
        RCCL_CHECK(ncclCommInitAll(comms_.data(), num_gpus_, nullptr));
    }

    ~RCCLCommunicator() {
        for(auto &comm : comms_){
            if(comm != nullptr){
                ncclResult_t res = ncclCommDestroy(comm);
                if(res != ncclSuccess){
                    std::cerr << "Failed to destroy RCCL communicator: " << ncclGetErrorString(res) << "\n";
                }
                comm = nullptr;
            }
        }
    }

    ncclComm_t getComm(int rank) const {
        assert(rank >= 0 && rank < num_gpus_);
        return comms_[rank];
    }

    // Disable copy semantics
    RCCLCommunicator(const RCCLCommunicator&) = delete;
    RCCLCommunicator& operator=(const RCCLCommunicator&) = delete;

    // Enable move semantics
    RCCLCommunicator(RCCLCommunicator&& other) noexcept : num_gpus_(other.num_gpus_), comms_(std::move(other.comms_)) {
        other.num_gpus_ = 0;
    }

    RCCLCommunicator& operator=(RCCLCommunicator&& other) noexcept {
        if(this != &other){
            // Destroy existing communicators
            for(auto &comm : comms_){
                if(comm != nullptr){
                    ncclCommDestroy(comm);
                }
            }
            num_gpus_ = other.num_gpus_;
            comms_ = std::move(other.comms_);
            other.num_gpus_ = 0;
        }
        return *this;
    }

private:
    int num_gpus_;
    std::vector<ncclComm_t> comms_;
};

// Structure to hold per-GPU resources
struct GPUResources {
    int rank;           // Unique rank assigned to the GPU
    int device_id;      // HIP device ID
    int actual_rows;    // Actual number of rows this GPU will process
    DeviceMemory* d_A = nullptr; // Device pointer for matrix A
    DeviceMemory* d_B = nullptr; // Device pointer for matrix B
    DeviceMemory* d_C = nullptr; // Device pointer for matrix C
    HIPStream* stream = nullptr;  // HIP stream
};

// Mutex for synchronizing console output
std::mutex cout_mutex;

// Function to handle each GPU's operations in a separate thread
void gpuWorker(int rank, GPUResources& gpu, 
               const std::vector<float>& h_A, const std::vector<float>& h_B, 
               int rows_per_gpu, int M, int N, int K, 
               std::vector<float>& h_C, 
               RCCLCommunicator& communicator) {
    try {
        // Set current GPU device
        HIP_CHECK(hipSetDevice(gpu.device_id));

        // Create a HIP stream for asynchronous operations
        HIPStream stream;
        gpu.stream = &stream;

        // Allocate device memory for A, B, and C using RAII
        DeviceMemory d_A(gpu.actual_rows * N * sizeof(float));
        DeviceMemory d_B(N * K * sizeof(float));
        DeviceMemory d_C(gpu.actual_rows * K * sizeof(float));

        gpu.d_A = &d_A;
        gpu.d_B = &d_B;
        gpu.d_C = &d_C;

        // Copy the relevant slice of A to device
        HIP_CHECK(hipMemcpyAsync(gpu.d_A->get(), 
                                 h_A.data() + rank * rows_per_gpu * N, 
                                 gpu.actual_rows * N * sizeof(float), 
                                 hipMemcpyHostToDevice, 
                                 stream.get()));

        // Ensure that rank 0 copies B before broadcasting
        if(rank == 0){
            // Copy B to device 0
            HIP_CHECK(hipMemcpyAsync(gpu.d_B->get(), 
                                     h_B.data(), 
                                     N * K * sizeof(float), 
                                     hipMemcpyHostToDevice, 
                                     stream.get()));
        }

        // Synchronize stream to ensure B is copied before broadcasting
        HIP_CHECK(hipStreamSynchronize(stream.get()));

        // Broadcast B from rank 0 to all other ranks
        RCCL_CHECK(ncclBroadcast(gpu.d_B->get(), 
                                 gpu.d_B->get(), 
                                 N * K, 
                                 ncclFloat, 
                                 0, 
                                 communicator.getComm(rank), 
                                 stream.get()));

        // Launch the matrix multiplication kernel
        dim3 blockDim(32, 32);
        dim3 gridDim((K + blockDim.x - 1) / blockDim.x, 
                    (gpu.actual_rows + blockDim.y - 1) / blockDim.y);

        hipLaunchKernelGGL(matMulKernel, gridDim, blockDim, 0, stream.get(),
                           gpu.d_A->get(), gpu.d_B->get(), gpu.d_C->get(), 
                           gpu.actual_rows, N, K);
        HIP_CHECK(hipGetLastError());

        // Synchronize stream to ensure kernel execution is complete
        HIP_CHECK(hipStreamSynchronize(stream.get()));

        // Copy the result back to host
        HIP_CHECK(hipMemcpyAsync(h_C.data() + rank * rows_per_gpu * K, 
                                 gpu.d_C->get(), 
                                 gpu.actual_rows * K * sizeof(float), 
                                 hipMemcpyDeviceToHost, 
                                 stream.get()));

        // Synchronize stream to ensure data is copied back
        HIP_CHECK(hipStreamSynchronize(stream.get()));

    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error in GPU " << rank << ": " << e.what() << "\n";
        // Depending on the design, you might want to rethrow or handle the error differently
        // For this example, we'll exit the program
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    try {
        // Matrix dimensions
        // C = A (M x N) * B (N x K)
        const int M = 2048; // Number of rows in A and C
        const int N = 2048; // Number of columns in A and rows in B
        const int K = 2048; // Number of columns in B and C

        // Seed for random number generation
        srand(static_cast<unsigned>(time(0)));

        // Initialize host matrices
        std::vector<float> h_A(M * N);
        std::vector<float> h_B(N * K);
        std::vector<float> h_C(M * K, 0.0f);      // Result from GPU
        std::vector<float> h_C_ref(M * K, 0.0f);  // Reference result

        // Initialize matrices with random values
        initializeMatrix(h_A);
        initializeMatrix(h_B);

        // Validate input sizes
        assert(h_A.size() == M * N);
        assert(h_B.size() == N * K);

        // Determine the number of available GPUs
        int num_gpus;
        HIP_CHECK(hipGetDeviceCount(&num_gpus));

        std::cout << "Number of GPUs available: " << num_gpus << "\n";

        if(num_gpus < 1){
            std::cerr << "No GPUs found.\n";
            return EXIT_FAILURE;
        }

        // Initialize communicators using RAII
        RCCLCommunicator communicator(num_gpus);

        // Calculate rows per GPU for load balancing
        int rows_per_gpu = (M + num_gpus - 1) / num_gpus; // Ceiling division

        // Initialize GPU resources
        std::vector<GPUResources> gpus_resources(num_gpus);
        for(int i = 0; i < num_gpus; ++i){
            gpus_resources[i].rank = i;
            gpus_resources[i].device_id = i;
            // Calculate actual_rows for each GPU
            if(i < num_gpus - 1){
                gpus_resources[i].actual_rows = rows_per_gpu;
            }
            else{
                // Last GPU may have fewer rows
                gpus_resources[i].actual_rows = M - rows_per_gpu * (num_gpus - 1);
                // Handle case where M is perfectly divisible
                if(gpus_resources[i].actual_rows <= 0){
                    gpus_resources[i].actual_rows = rows_per_gpu;
                }
            }
            // Ensure actual_rows is positive
            assert(gpus_resources[i].actual_rows > 0 && "Each GPU must process at least one row.");
        }

        // Start timing the CPU reference
        std::cout << "Computing reference result on CPU...\n";
        auto cpu_start = std::chrono::high_resolution_clock::now();
        matMulCPU(h_A, h_B, h_C_ref, M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
        std::cout << "CPU matrix multiplication took " << cpu_duration.count() << " seconds.\n";

        // Start timing the GPU implementation
        std::cout << "Computing matrix multiplication on GPUs...\n";
        auto gpu_start = std::chrono::high_resolution_clock::now();

        // Launch a thread for each GPU to handle communication and computation
        std::vector<std::thread> threads;
        for(int i = 0; i < num_gpus; ++i){
            threads.emplace_back(gpuWorker, i, std::ref(gpus_resources[i]), 
                                 std::cref(h_A), std::cref(h_B), 
                                 rows_per_gpu, M, N, K, 
                                 std::ref(h_C), 
                                 std::ref(communicator));
        }

        // Wait for all threads to complete
        for(auto &t : threads){
            if(t.joinable()){
                t.join();
            }
        }

        // End timing the GPU implementation
        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
        std::cout << "GPU matrix multiplication took " << gpu_duration.count() << " seconds.\n";
        std::cout << "Speed-up: " << cpu_duration.count() / gpu_duration.count() << "x\n";

        // Validate the GPU result against the CPU result
        bool valid = true;
        const float epsilon = 1e-3f;
        for(int i = 0; i < M * K; ++i){
            if(std::abs(h_C_ref[i] - h_C[i]) > epsilon){
                std::cerr << "Mismatch at index " << i << ": CPU=" 
                          << h_C_ref[i] << ", GPU=" << h_C[i] << "\n";
                valid = false;
                break;
            }
        }

        if(valid){
            std::cout << "Validation PASSED: GPU result matches CPU result.\n";
        }
        else{
            std::cerr << "Validation FAILED: GPU result does not match CPU result.\n";
        }

        return valid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& e){
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}

