#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to perform element-wise vector addition
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Size of vectors
    int N = 1 << 20;  // For example, 2^20 elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define the number of threads in a block
    int blockSize = 256;
    // Define the number of blocks in a grid
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch the vecAdd CUDA kernel
    vecAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copy the result vector from device memory to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            cerr << "Error at index " << i << ": " <<
                h_C[i] << " != " << h_A[i] << " + " << h_B[i] << endl;
            success = false;
            break;
        }
    }
    if (success) {
        cout << "Vector addition successful!" << endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}