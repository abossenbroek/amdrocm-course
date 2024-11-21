// Include necessary headers
#include <hip/hip_runtime.h> // HIP runtime API
#include <iostream>         // For std::cout and std::abort
#include <vector>           // For std::vector
#include <chrono>	    // For std::chrono

// Define the HIP_CHECK macro for error handling
#define CHECK_RET_CODE(call, ret_code)                                                             \
  {                                                                                                \
    if ((call) != ret_code) {                                                                      \
      std::cout << "Failed in call: " << #call << std::endl;                                       \
      std::abort();                                                                                \
    }                                                                                              \
  }
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)

__global__ void vectorVectorAdd(const float* __restrict__ input_A, const float* __restrict__ input_B, float* __restrict__ output, size_t size) {
    // Calculate the global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't access out-of-bounds memory
    if (idx < size) {
        // Perform the vector addition
        output[idx] = input_A[idx] + input_B[idx];
    }
}

	

int main() {
    // Size of the vector
    size_t N = 1 << 24; // 268M elements

    // Initialize host input vector with some values
    std::vector<float> a_input(N, 1.0f);  // Initialize all elements to 1.0
    std::vector<float> b_input(N, 2.5f);  // Initialize all elements to 2.5
    std::vector<float> h_output(N, 0.0f); // Initialize output vector to 0.0

    // Device pointers
    float *a_d_input = nullptr;
    float *b_d_input = nullptr;
    float *d_output = nullptr;

    // Allocate device memory for input vector A
    HIP_CHECK(hipMalloc(&a_d_input, N * sizeof(float))); // Allocate memory on the device
    // Allocate device memory for input vector B
    HIP_CHECK(hipMalloc(&b_d_input, N * sizeof(float))); // Allocate memory on the device
    // Allocate device memory for output vector
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float))); // Allocate memory on the device

    // Copy input data A from host to device
    HIP_CHECK(hipMemcpy(a_d_input, a_input.data(), N * sizeof(float), hipMemcpyHostToDevice)); // Transfer input data to device
    // Copy input data B from host to device
    HIP_CHECK(hipMemcpy(b_d_input, b_input.data(), N * sizeof(float), hipMemcpyHostToDevice)); // Transfer input data to device

    // Define block size and grid size for the kernel launch
    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed
  
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the vector-scalar multiplication kernel
    hipLaunchKernelGGL(vectorVectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, a_d_input, b_d_input, d_output, N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu = end - start;
    std::cout << "GPU compute time: " << elapsed_gpu.count() << " seconds\n";

    // Check for any errors during kernel launch or execution
    HIP_CHECK(hipGetLastError()); // Retrieve and check the last error

    // Copy the result from device back to host
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, N * sizeof(float), hipMemcpyDeviceToHost)); // Transfer output data to host

    // (Optional) Verify the results
    bool success = true;
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        if (h_output[i] != a_input[i] + b_input[i]) { // Check if each element is correctly multiplied
            std::cout << "Mismatch at index " << i << ": " << h_output[i] << " != " << a_input[i] + b_input[i] << std::endl;
            success = false;
            break;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end - start;
    std::cout << "CPU verification time: " << elapsed_cpu.count() << " seconds\n";
    std::cout << "Speed-up: " << elapsed_cpu.count() / elapsed_gpu.count() << std::endl;

    if (success) {
        std::cout << "Vector-vector multiplication successful!" << std::endl; // Success message
    }

    // Free device memory for input vector A
    HIP_CHECK(hipFree(a_d_input)); // Deallocate device memory
    // Free device memory for input vector B
    HIP_CHECK(hipFree(b_d_input)); // Deallocate device memory
    // Free device memory for output vector
    HIP_CHECK(hipFree(d_output)); // Deallocate device memory

    return 0; // Exit the program
}
