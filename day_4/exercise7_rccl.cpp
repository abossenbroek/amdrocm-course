/**
 * @file main.cpp
 * @brief Demonstrates collective communication using RCCL over MPI in a multi-process HIP application.
 *
 * This program initializes multiple GPU workers, each on a different device (GPU),
 * and performs collective operations like broadcast, all-reduce, and all-gather using RCCL.
 * MPI is used to initialize RCCL communicators across multiple nodes.
 * The code includes extensive documentation for educational purposes.
 */

#include <cassert>
#include <functional>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mpi.h>
#include <numeric> // For std::iota
#include <random>
#include <rccl/rccl.h>
#include <stdexcept>
#include <vector>

// Macros for error checking
/**
 * @brief Checks the result of a HIP API call and throws an exception on error.
 *
 * @param cmd The HIP API function call.
 */
#define HIP_CHECK(cmd)                                                                                                 \
    {                                                                                                                  \
        hipError_t error = cmd;                                                                                        \
        if (error != hipSuccess)                                                                                       \
        {                                                                                                              \
            throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(error) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                        \
        }                                                                                                              \
    }

/**
 * @brief Checks the result of an RCCL API call and throws an exception on error.
 *
 * @param cmd The RCCL API function call.
 */
#define RCCL_CHECK(cmd)                                                                                                \
    {                                                                                                                  \
        ncclResult_t error = cmd;                                                                                      \
        if (error != ncclSuccess)                                                                                      \
        {                                                                                                              \
            throw std::runtime_error(std::string("RCCL Error: ") + ncclGetErrorString(error) + " at " + __FILE__ +     \
                                     ":" + std::to_string(__LINE__));                                                  \
        }                                                                                                              \
    }

/**
 * @brief Retrieves a human-readable error string for an RCCL error code.
 *
 * @param error The RCCL error code.
 * @return const char* The error string.
 */
const char* ncclGetErrorString(ncclResult_t error)
{
    switch (error)
    {
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

/**
 * @class HIPStream
 * @brief RAII wrapper class for managing a HIP stream.
 *
 * This class encapsulates the creation and destruction of a HIP stream,
 * ensuring that resources are properly released when the object goes out of scope.
 */
class HIPStream
{
public:
    /**
     * @brief Constructs a HIPStream object and creates a HIP stream.
     */
    HIPStream()
    {
        HIP_CHECK(hipStreamCreate(&stream_));
    }

    /**
     * @brief Destructs the HIPStream object and destroys the HIP stream.
     */
    ~HIPStream()
    {
        if (stream_)
        {
            hipError_t err = hipStreamDestroy(stream_);
            if (err != hipSuccess)
            {
                std::cerr << "Failed to destroy HIP stream: " << hipGetErrorString(err) << "\n";
            }
            stream_ = nullptr;
        }
    }

    /**
     * @brief Retrieves the HIP stream handle.
     *
     * @return hipStream_t The HIP stream handle.
     */
    hipStream_t get() const
    {
        return stream_;
    }

private:
    hipStream_t stream_ = nullptr; /**< The HIP stream handle */
};

/**
 * @class DeviceMemory
 * @brief RAII wrapper class for managing device memory allocations.
 *
 * This class handles the allocation and deallocation of device memory,
 * ensuring that memory is freed when the object goes out of scope.
 */
class DeviceMemory
{
public:
    /**
     * @brief Constructs a DeviceMemory object and allocates device memory.
     *
     * @param size The size of the memory to allocate, in bytes.
     */
    DeviceMemory(size_t size) : size_(size)
    {
        HIP_CHECK(hipMalloc(&ptr_, size_));
    }

    /**
     * @brief Destructs the DeviceMemory object and frees the device memory.
     */
    ~DeviceMemory()
    {
        if (ptr_)
        {
            hipError_t err = hipFree(ptr_);
            if (err != hipSuccess)
            {
                std::cerr << "Failed to free device memory: " << hipGetErrorString(err) << "\n";
            }
            ptr_ = nullptr;
        }
    }

    /**
     * @brief Retrieves the pointer to the device memory.
     *
     * @return float* Pointer to the allocated device memory.
     */
    float* get() const
    {
        return ptr_;
    }

    // Disable copy semantics
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Enable move semantics
    /**
     * @brief Move constructor for DeviceMemory.
     *
     * @param other The DeviceMemory object to move from.
     */
    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator for DeviceMemory.
     *
     * @param other The DeviceMemory object to move from.
     * @return DeviceMemory& Reference to the assigned object.
     */
    DeviceMemory& operator=(DeviceMemory&& other) noexcept
    {
        if (this != &other)
        {
            if (ptr_)
            {
                hipError_t err = hipFree(ptr_);
                if (err != hipSuccess)
                {
                    std::cerr << "Failed to free device memory: " << hipGetErrorString(err) << "\n";
                }
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    float* ptr_ = nullptr; /**< Pointer to the device memory */
    size_t size_ = 0;      /**< Size of the allocated memory */
};

/**
 * @brief Kernel function to add two arrays element-wise.
 *
 * @param A Pointer to the first input array and output array.
 * @param B Pointer to the second input array.
 * @param size The number of elements in the arrays.
 */
__global__ void add_arrays(float* A, float* B, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        A[tid] += B[tid];
    }
}

/**
 * @brief Kernel function to compute the mean squared error (MSE) of an array.
 *
 * @param A Pointer to the input array.
 * @param mse Pointer to the output scalar MSE value.
 * @param size The number of elements in the array.
 */
__global__ void compute_mse(float* A, float* mse, int size)
{
    __shared__ float shared_mse[256]; // Shared memory for block-wise reduction
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;

    float local_mse = 0.0f;
    if (tid < size)
    {
        local_mse = A[tid] * A[tid];
    }

    // Store the local MSE in shared memory
    shared_mse[lane] = local_mse;
    __syncthreads();

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            shared_mse[lane] += shared_mse[lane + stride];
        }
        __syncthreads();
    }

    // The first thread in the block adds the block's result to the global MSE
    if (lane == 0)
    {
        atomicAdd(mse, shared_mse[0] / size);
    }
}

/**
 * @class GPUWorker
 * @brief Represents a worker that performs computations on a GPU device.
 *
 * This class encapsulates the initialization of device-specific resources,
 * the execution of computational kernels, and the use of RCCL for collective communication.
 */
class GPUWorker
{
public:
    /**
     * @brief Constructs a GPUWorker object.
     *
     * @param rank The rank of the worker (GPU device index).
     * @param world_size The total number of workers.
     * @param comm The RCCL communicator.
     */
    GPUWorker(int rank, int world_size, ncclComm_t comm) : rank_(rank), world_size_(world_size), comm_(comm)
    {
        // Initialize host data
        h_A_.resize(ARRAY_SIZE_);
        h_B_.resize(ARRAY_SIZE_);
        h_C_.resize(256 * world_size_);

        // Initialize random number generator with a different seed per rank
        std::random_device rd;
        std::mt19937 gen(rd() + rank_);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (rank_ == 0)
        {
            // Rank 0 initializes arrays A, B, and C
            for (auto& val : h_A_)
                val = dis(gen);
            std::fill(h_B_.begin(), h_B_.end(), 1.0f);
            std::iota(h_C_.begin(), h_C_.begin() + 256, 1.0f);
        }
        else
        {
            // Other ranks initialize array A and partial array C
            for (auto& val : h_A_)
                val = dis(gen);
            std::fill(h_C_.begin() + 256 * rank_, h_C_.begin() + 256 * (rank_ + 1), static_cast<float>(rank_));
        }

        // Print initialization message
        std::cout << "Rank " << rank_ << " initialized data\n";
    }

    /**
     * @brief Executes the main computation and communication tasks.
     *
     * This method performs the following steps:
     * - Sets the device context.
     * - Allocates device memory and copies data from host to device.
     * - Performs collective communications (broadcast, all-reduce, all-gather) using RCCL.
     * - Executes computational kernels (addition, MSE computation).
     * - Synchronizes and cleans up resources.
     */
    void run()
    {
        // Set the device for this worker
        HIP_CHECK(hipSetDevice(rank_));

        // Initialize the HIP stream
        stream_ = std::make_unique<HIPStream>();

        // Allocate device memory
        d_A_ = std::make_unique<DeviceMemory>(ARRAY_SIZE_ * sizeof(float));
        d_B_ = std::make_unique<DeviceMemory>(ARRAY_SIZE_ * sizeof(float));
        d_C_ = std::make_unique<DeviceMemory>(256 * world_size_ * sizeof(float));
        d_mse_ = std::make_unique<DeviceMemory>(sizeof(float));

        // Copy host data to device asynchronously
        HIP_CHECK(hipMemcpyAsync(d_A_->get(), h_A_.data(), ARRAY_SIZE_ * sizeof(float), hipMemcpyHostToDevice,
                                 stream_->get()));
        HIP_CHECK(hipMemcpyAsync(d_B_->get(), h_B_.data(), ARRAY_SIZE_ * sizeof(float), hipMemcpyHostToDevice,
                                 stream_->get()));
        HIP_CHECK(hipMemcpyAsync(d_C_->get(), h_C_.data(), 256 * world_size_ * sizeof(float), hipMemcpyHostToDevice,
                                 stream_->get()));

        // Synchronize the stream to ensure data is copied before proceeding
        HIP_CHECK(hipStreamSynchronize(stream_->get()));

        std::cout << "Rank " << rank_ << ": Entered run() method\n";

        // Perform a broadcast operation to distribute B from rank 0 to all other ranks
        std::cout << "Rank " << rank_ << ": Starting ncclBroadcast\n";
        RCCL_CHECK(ncclBroadcast(d_B_->get(), d_B_->get(), ARRAY_SIZE_, ncclFloat, 0, comm_, stream_->get()));
        HIP_CHECK(hipStreamSynchronize(stream_->get()));
        std::cout << "Rank " << rank_ << ": Completed ncclBroadcast\n";

        // Launch the kernel to add arrays A and B element-wise
        add_arrays<<<GRID_SIZE_, BLOCK_SIZE_, 0, stream_->get()>>>(d_A_->get(), d_B_->get(), ARRAY_SIZE_);
        HIP_CHECK(hipStreamSynchronize(stream_->get()));
        std::cout << "Rank " << rank_ << ": Completed add_arrays kernel\n";

        // Initialize the MSE value on device to zero
        HIP_CHECK(hipMemsetAsync(d_mse_->get(), 0, sizeof(float), stream_->get()));

        // Launch the kernel to compute the mean squared error (MSE)
        compute_mse<<<GRID_SIZE_, BLOCK_SIZE_, 0, stream_->get()>>>(d_A_->get(), d_mse_->get(), ARRAY_SIZE_);
        HIP_CHECK(hipStreamSynchronize(stream_->get()));
        std::cout << "Rank " << rank_ << ": Completed compute_mse kernel\n";

        // Perform an all-reduce operation to sum the MSE values across all ranks
        std::cout << "Rank " << rank_ << ": Starting ncclAllReduce\n";
        RCCL_CHECK(ncclAllReduce(d_mse_->get(), d_mse_->get(), 1, ncclFloat, ncclSum, comm_, stream_->get()));
        HIP_CHECK(hipStreamSynchronize(stream_->get()));
        std::cout << "Rank " << rank_ << ": Completed ncclAllReduce\n";

        // Copy the global MSE value back to the host
        HIP_CHECK(hipMemcpy(&global_mse_, d_mse_->get(), sizeof(float), hipMemcpyDeviceToHost));
        std::cout << "Rank " << rank_ << ": Global MSE value after AllReduce: " << global_mse_ << "\n";

        // Perform an all-gather operation on array C
        std::cout << "Rank " << rank_ << ": Starting ncclAllGather\n";
        RCCL_CHECK(ncclAllGather(d_C_->get() + 256 * rank_, d_C_->get(), 256, ncclFloat, comm_, stream_->get()));
        HIP_CHECK(hipStreamSynchronize(stream_->get()));
        std::cout << "Rank " << rank_ << ": Completed ncclAllGather\n";

        // Optionally, only rank 0 prints the final results
        if (rank_ == 0)
        {
            // Copy the gathered array C back to the host
            HIP_CHECK(hipMemcpy(h_C_.data(), d_C_->get(), 256 * world_size_ * sizeof(float), hipMemcpyDeviceToHost));
            std::cout << "Rank 0: First few elements of h_C_ after AllGather:\n";
            for (int i = 0; i < 10; ++i)
            {
                std::cout << h_C_[i] << " ";
            }
            std::cout << "\n";
            std::cout << "Rank 0: Final Global MSE value: " << global_mse_ << "\n";
        }

        std::cout << "Rank " << rank_ << ": Exiting run() method\n";

        // Clean up resources
        // Destroy the RCCL communicator (done in main)
    }

    // Disable copy semantics
    GPUWorker(const GPUWorker&) = delete;
    GPUWorker& operator=(const GPUWorker&) = delete;

private:
    int rank_;                            /**< The rank of the worker (GPU device index) */
    int world_size_;                      /**< The total number of workers */
    ncclComm_t comm_ = nullptr;           /**< The RCCL communicator handle */
    std::unique_ptr<HIPStream> stream_;   /**< The HIP stream used for GPU operations */
    std::unique_ptr<DeviceMemory> d_A_;   /**< Device memory for array A */
    std::unique_ptr<DeviceMemory> d_B_;   /**< Device memory for array B */
    std::unique_ptr<DeviceMemory> d_C_;   /**< Device memory for array C */
    std::unique_ptr<DeviceMemory> d_mse_; /**< Device memory for the MSE value */
    std::vector<float> h_A_;              /**< Host memory for array A */
    std::vector<float> h_B_;              /**< Host memory for array B */
    std::vector<float> h_C_;              /**< Host memory for array C */
    float global_mse_ = 0.0f;             /**< The global MSE value after all-reduce */

    // Constants
    const int ARRAY_SIZE_ = 1 << 10; /**< The number of elements in the arrays (2^10) */
    const int BLOCK_SIZE_ = 256;     /**< The number of threads per block in GPU kernels */
    const int GRID_SIZE_ = (ARRAY_SIZE_ + BLOCK_SIZE_ - 1) / BLOCK_SIZE_; /**< The number of blocks in the grid */
};

/**
 * @brief The main function initializes the application and launches GPU worker processes.
 *
 * @param argc The argument count.
 * @param argv The argument vector.
 * @return int Exit code indicating success or failure.
 */
int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize HIP and get the number of available GPUs
    int num_gpus;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));

    if (num_gpus < 1)
    {
        std::cerr << "No GPUs found.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (world_size > num_gpus)
    {
        std::cerr << "Number of MPI processes (" << world_size << ") exceeds number of GPUs (" << num_gpus << ")\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Each process selects its own GPU device
    HIP_CHECK(hipSetDevice(rank));

    // Generate a unique RCCL ID on rank 0 and broadcast it to all others
    ncclUniqueId id;
    if (rank == 0)
    {
        RCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize RCCL communicator
    ncclComm_t comm;
    RCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));

    // Each process creates a GPUWorker and runs it
    {
        GPUWorker worker(rank, world_size, comm);
        worker.run();
    }

    // Clean up RCCL communicator
    RCCL_CHECK(ncclCommDestroy(comm));

    MPI_Finalize();
    return 0;
}
