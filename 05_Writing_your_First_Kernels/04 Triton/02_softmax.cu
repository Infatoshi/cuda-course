#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void softmax_cuda(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float max_val = input[tid];
        for (int i = tid + n; i < n * n; i += n) {
            max_val = max(max_val, input[i]);
        }
        
        float sum = 0.0f;
        for (int i = tid; i < n * n; i += n) {
            sum += expf(input[i] - max_val);
        }
        
        for (int i = tid; i < n * n; i += n) {
            output[i] = expf(input[i] - max_val) / sum;
        }
    }
}

void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

int main() {
    const int n = 1000;  // Increased vector size
    float *x_cpu = (float*)malloc(n * sizeof(float));
    float *x_gpu = (float*)malloc(n * sizeof(float));
    float *d_input, *d_output;

    // Initialize input vector
    for (int i = 0; i < n; i++) {
        x_cpu[i] = (float)rand() / RAND_MAX;  // Random values between 0 and 1
        x_gpu[i] = x_cpu[i];  // Copy to GPU input
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, x_gpu, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    softmax_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    // Copy result back to host
    cudaMemcpy(x_gpu, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute softmax on CPU
    softmax(x_cpu, n);

    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(x_cpu[i] - x_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("Maximum difference between CPU and GPU results: %e\n", max_diff);

    // Clean up
    free(x_cpu);
    free(x_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}