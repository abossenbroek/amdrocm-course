#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <hip/hip_runtime.h>

// Constants
constexpr int TILE_SIZE = 16;  // Tile size for the matrix multiplication

// Define the HIP_CHECK macro for error handling
#define CHECK_RET_CODE(call, ret_code)                                                             \
  {                                                                                                \
    if ((call) != ret_code) {                                                                      \
      std::cout << "Failed in call: " << #call << std::endl;                                       \
      std::abort();                                                                                \
    }                                                                                              \
  }
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)

// Function to initialize matrices with random values
void initializeMatrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
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
    Matrix(int rows, int cols)
        : m_rows(rows), m_cols(cols), m_size(rows * cols * sizeof(float)) {
        HIP_CHECK(hipMalloc(&m_data, m_size));  // Using HIP_CHECK macro for error handling
    }

    ~Matrix() {
        if (m_data) {
            HIP_CHECK(hipFree(m_data));  // Using HIP_CHECK macro for error handling
        }
    }

    void uploadData(const std::vector<float>& data) {
        HIP_CHECK(hipMemcpy(m_data, data.data(), m_size, hipMemcpyHostToDevice));  // Using HIP_CHECK macro for error handling
    }

    void downloadData(std::vector<float>& data) const {
        HIP_CHECK(hipMemcpy(data.data(), m_data, m_size, hipMemcpyDeviceToHost));  // Using HIP_CHECK macro for error handling
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
    const int N = 1024;  // Matrix size (N x N)

    // Host matrices
    std::vector<float> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);

    // Initialize matrices
    initializeMatrix(A, N, N);
    initializeMatrix(B, N, N);

    // Perform CPU multiplication
    matMulCPU(A, B, C_cpu, N);

    // GPU setup
    Matrix d_A(N, N), d_B(N, N), d_C(N, N);

    d_A.uploadData(A);
    d_B.uploadData(B);

    // Create a stream for asynchronous execution
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));  // Using HIP_CHECK macro for error handling

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Perform GPU multiplication using the stream
    matMulGPU<<<gridSize, blockSize, 0, stream>>>(d_A.data(), d_B.data(), d_C.data(), N);
    HIP_CHECK(hipGetLastError());  // Using HIP_CHECK macro for error handling

    // Synchronize the stream to ensure completion of the kernel
    HIP_CHECK(hipStreamSynchronize(stream));  // Using HIP_CHECK macro for error handling

    // Download the result from GPU
    d_C.downloadData(C_gpu);

    // Check if the results match
    if (compareResults(C_cpu, C_gpu, N)) {
        std::cout << "CPU and GPU results match!" << std::endl;
    } else {
        std::cerr << "Results do not match!" << std::endl;
    }

    // Destroy the stream
    HIP_CHECK(hipStreamDestroy(stream));  // Using HIP_CHECK macro for error handling

    return 0;
}

