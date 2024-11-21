#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <hip/hip_runtime.h>
#include <random>

// Constants
constexpr int TILE_SIZE = 16;      // Tile size for the matrix multiplication
constexpr int NUM_STREAMS = 4;     // Number of streams for parallel execution
constexpr int MATRIX_SIZE = 1024;  // Matrix size (N x N)

// Define the HIP_CHECK macro for error handling
#define CHECK_RET_CODE(call, ret_code)                                                             \
  {                                                                                                \
    if ((call) != ret_code) {                                                                      \
      std::cout << "Failed in call: " << #call << " with error code " << (call) << std::endl;     \
      std::abort();                                                                                \
    }                                                                                              \
  }
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)

// Function to initialize matrices with random values using a given seed
void initializeMatrix(std::vector<float>& mat, int rows, int cols, unsigned int seed) {
    std::mt19937 gen(seed);  // Initialize random number generator with the seed
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // Uniform distribution [0, 1)

    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dist(gen);  // Fill matrix with random values based on the seed
    }
}

// CPU implementation of matrix multiplication
void matMulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// GPU kernel for matrix multiplication with tiling
__global__ void matMulGPU(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

// RAII class for handling matrix memory on the GPU
class Matrix {
public:
    // Delete copy constructor and copy assignment operator to prevent copying
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Implement move constructor and move assignment operator
    Matrix(Matrix&& other) noexcept
        : m_rows(other.m_rows), m_cols(other.m_cols), m_size(other.m_size), m_data(other.m_data) {
        other.m_data = nullptr;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            if (m_data) {
                HIP_CHECK(hipFree(m_data));
            }
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_size = other.m_size;
            m_data = other.m_data;
            other.m_data = nullptr;
        }
        return *this;
    }

    Matrix(int rows, int cols)
        : m_rows(rows), m_cols(cols), m_size(rows * cols * sizeof(float)), m_data(nullptr) {
        HIP_CHECK(hipMalloc(&m_data, m_size));  // Using HIP_CHECK macro for error handling
    }

    ~Matrix() {
        if (m_data) {
            // Use a separate error check to avoid aborting in destructor
            hipError_t err = hipFree(m_data);
            if (err != hipSuccess) {
                std::cerr << "Failed to free GPU memory in destructor: " << hipGetErrorString(err) << std::endl;
            }
        }
    }

    void uploadDataAsync(const std::vector<float>& data, hipStream_t stream) {
        HIP_CHECK(hipMemcpyAsync(m_data, data.data(), m_size, hipMemcpyHostToDevice, stream));  // Async upload
    }

    void downloadDataAsync(std::vector<float>& data, hipStream_t stream) const {
        HIP_CHECK(hipMemcpyAsync(data.data(), m_data, m_size, hipMemcpyDeviceToHost, stream));  // Async download
    }

    float* data() { return m_data; }
    const float* data() const { return m_data; }

    int rows() const { return m_rows; }
    int cols() const { return m_cols; }

private:
    int m_rows, m_cols;
    size_t m_size;
    float* m_data;
};

// RAII class for stream management
class Stream {
public:
    Stream() : m_stream(nullptr) {
        HIP_CHECK(hipStreamCreate(&m_stream));  // Create the stream and check for errors
    }

    // Delete copy constructor and copy assignment operator
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Implement move constructor and move assignment operator
    Stream(Stream&& other) noexcept : m_stream(other.m_stream) {
        other.m_stream = nullptr;
    }

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            if (m_stream) {
                HIP_CHECK(hipStreamDestroy(m_stream));
            }
            m_stream = other.m_stream;
            other.m_stream = nullptr;
        }
        return *this;
    }

    ~Stream() {
        if (m_stream) {
            // Use a separate error check to avoid aborting in destructor
            hipError_t err = hipStreamDestroy(m_stream);
            if (err != hipSuccess) {
                std::cerr << "Failed to destroy HIP stream in destructor: " << hipGetErrorString(err) << std::endl;
            }
        }
    }

    hipStream_t get() const { return m_stream; }

private:
    hipStream_t m_stream;
};

// Function to check if CPU and GPU results are equal
bool compareResults(const std::vector<float>& cpuResult, const std::vector<float>& gpuResult, int N, float tolerance = 1e-4f) {
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(cpuResult[i] - gpuResult[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": CPU = " << cpuResult[i] << ", GPU = " << gpuResult[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    unsigned int seed = 12345; // Set a seed for deterministic matrix initialization

    std::cout << "Initializing host matrices..." << std::endl;

    // Host matrices for each stream
    std::vector<std::vector<float>> hostA(NUM_STREAMS, std::vector<float>(MATRIX_SIZE * MATRIX_SIZE));
    std::vector<std::vector<float>> hostB(NUM_STREAMS, std::vector<float>(MATRIX_SIZE * MATRIX_SIZE));
    std::vector<std::vector<float>> C_cpu(NUM_STREAMS, std::vector<float>(MATRIX_SIZE * MATRIX_SIZE)); // For validation

    // Initialize host matrices with unique seeds for each stream
    for (int i = 0; i < NUM_STREAMS; ++i) {
        initializeMatrix(hostA[i], MATRIX_SIZE, MATRIX_SIZE, seed + i);
        initializeMatrix(hostB[i], MATRIX_SIZE, MATRIX_SIZE, seed + NUM_STREAMS + i);
    }

    std::cout << "Host matrices initialized." << std::endl;

    // Initialize streams using RAII
    std::cout << "Creating HIP streams..." << std::endl;
    std::vector<Stream> streams;
    streams.reserve(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        streams.emplace_back();
        std::cout << "Stream " << i << " created." << std::endl;
    }

    // Create matrices on the GPU for each stream: A_i, B_i, C_i
    struct GPU_Matrices {
        Matrix A;
        Matrix B;
        Matrix C;
        GPU_Matrices(int N) : A(N, N), B(N, N), C(N, N) {}
    };

    std::cout << "Allocating GPU matrices for each stream..." << std::endl;
    std::vector<GPU_Matrices> gpuMatrices;
    gpuMatrices.reserve(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        gpuMatrices.emplace_back(MATRIX_SIZE);
        std::cout << "GPU matrices for Stream " << i << " allocated." << std::endl;
    }

    // Upload A_i and B_i to GPU asynchronously on respective streams
    std::cout << "Uploading matrices to GPU..." << std::endl;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        std::cout << "Uploading A_" << i << " and B_" << i << " to GPU on Stream " << i << "..." << std::endl;
        gpuMatrices[i].A.uploadDataAsync(hostA[i], streams[i].get());
        gpuMatrices[i].B.uploadDataAsync(hostB[i], streams[i].get());
    }
    std::cout << "All matrices uploaded to GPU." << std::endl;

    // Synchronize all streams to ensure all operations are complete
    std::cout << "Synchronizing all streams..." << std::endl;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        HIP_CHECK(hipStreamSynchronize(streams[i].get()));
        std::cout << "Stream " << i << " synchronized." << std::endl;
    }
    std::cout << "All streams synchronized." << std::endl;

    // Launch GPU kernels for each stream
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Launching GPU kernels..." << std::endl;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        std::cout << "Launching kernel for Stream " << i << "..." << std::endl;
        matMulGPU<<<grid, block, 0, streams[i].get()>>>(
            gpuMatrices[i].A.data(),
            gpuMatrices[i].B.data(),
            gpuMatrices[i].C.data(),
            MATRIX_SIZE
        );

        // Check for kernel launch errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed for Stream " << i << ": " << hipGetErrorString(err) << std::endl;
            std::abort();
        }
    }
    std::cout << "All GPU kernels launched." << std::endl;

    // Download C_i from GPU asynchronously on respective streams
    std::cout << "Downloading results from GPU..." << std::endl;
    std::vector<std::vector<float>> hostC_gpu(NUM_STREAMS, std::vector<float>(MATRIX_SIZE * MATRIX_SIZE));
    for (int i = 0; i < NUM_STREAMS; ++i) {
        std::cout << "Downloading C_" << i << " from GPU on Stream " << i << "..." << std::endl;
        gpuMatrices[i].C.downloadDataAsync(hostC_gpu[i], streams[i].get());
    }

    // Download A_i and B_i from GPU asynchronously for CPU validation
    std::cout << "Downloading A and B matrices from GPU for CPU validation..." << std::endl;
    std::vector<std::vector<float>> hostA_downloaded(NUM_STREAMS, std::vector<float>(MATRIX_SIZE * MATRIX_SIZE));
    std::vector<std::vector<float>> hostB_downloaded(NUM_STREAMS, std::vector<float>(MATRIX_SIZE * MATRIX_SIZE));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        std::cout << "Downloading A_" << i << " and B_" << i << " from GPU on Stream " << i << "..." << std::endl;
        gpuMatrices[i].A.downloadDataAsync(hostA_downloaded[i], streams[i].get());
        gpuMatrices[i].B.downloadDataAsync(hostB_downloaded[i], streams[i].get());
    }

    // Synchronize all streams to ensure all operations are complete
    std::cout << "Synchronizing all streams..." << std::endl;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        HIP_CHECK(hipStreamSynchronize(streams[i].get()));
        std::cout << "Stream " << i << " synchronized." << std::endl;
    }
    std::cout << "All streams synchronized." << std::endl;

    // Perform CPU multiplication and compare results for each stream
    std::cout << "Validating GPU results with CPU computations..." << std::endl;
    bool allMatch = true;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        std::cout << "Validating Stream " << i << "..." << std::endl;
        matMulCPU(hostA_downloaded[i], hostB_downloaded[i], C_cpu[i], MATRIX_SIZE);
        if (compareResults(C_cpu[i], hostC_gpu[i], MATRIX_SIZE)) {
            std::cout << "Stream " << i << " results match CPU!" << std::endl;
        } else {
            std::cerr << "Stream " << i << " results do not match CPU!" << std::endl;
            allMatch = false;
        }
    }

    if (allMatch) {
        std::cout << "All streams validated successfully. GPU results match CPU results." << std::endl;
    } else {
        std::cerr << "Validation failed for one or more streams." << std::endl;
    }

    return 0;
}
