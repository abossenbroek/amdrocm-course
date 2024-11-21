// Include necessary headers
#include <hip/hip_runtime.h> // HIP runtime API
#include <iostream>         // For std::cout and std::abort
#include <vector>           // For std::vector

// Define the HIP_CHECK macro for error handling
#define CHECK_RET_CODE(call, ret_code)                                                             \
  {                                                                                                \
    if ((call) != ret_code) {                                                                      \
      std::cout << "Failed in call: " << #call << std::endl;                                       \
      std::abort();                                                                                \
    }                                                                                              \
  }
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)

// Define the vector-scalar multiplication kernel
__global__ void vectorScalarMul(const float* __restrict__ input, float scalar, float* __restrict__ output, size_t size) {
    // Calculate the global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't access out-of-bounds memory
    if (idx < size) {
        // Perform the scalar multiplication
        output[idx] = input[idx] * scalar;
    }
}

int main() {
    // Size of the vector
    size_t N = 1 << 20; // 1M elements

    // Scalar value to multiply
    float scalar = 2.5f;

    // Initialize host input vector with some values
    std::vector<float> h_input(N, 1.0f);  // Initialize all elements to 1.0
    std::vector<float> h_output(N, 0.0f); // Initialize output vector to 0.0

    // Device pointers
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Allocate device memory for input vector
    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float))); // Allocate memory on the device
    // Allocate device memory for output vector
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float))); // Allocate memory on the device

    // Copy input data from host to device
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice)); // Transfer input data to device

    // Define block size and grid size for the kernel launch
    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed

    // Launch the vector-scalar multiplication kernel
    hipLaunchKernelGGL(vectorScalarMul, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_input, scalar, d_output, N);
    
    // Check for any errors during kernel launch or execution
    HIP_CHECK(hipGetLastError()); // Retrieve and check the last error

    // Copy the result from device back to host
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, N * sizeof(float), hipMemcpyDeviceToHost)); // Transfer output data to host

    // (Optional) Verify the results
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_output[i] != h_input[i] * scalar) { // Check if each element is correctly multiplied
            std::cout << "Mismatch at index " << i << ": " << h_output[i] << " != " << h_input[i] * scalar << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector-scalar multiplication successful!" << std::endl; // Success message
    }

    // Free device memory for input vector
    HIP_CHECK(hipFree(d_input)); // Deallocate device memory
    // Free device memory for output vector
    HIP_CHECK(hipFree(d_output)); // Deallocate device memory

    return 0; // Exit the program
}
