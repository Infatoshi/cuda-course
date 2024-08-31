#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void softmax_cuda(float* input, float* output, int B, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.y;
    
    if (tid < N && bid < B) {
        int offset = bid * N;
        float max_val = input[offset + tid];
        for (int i = 1; i < N; i++) {
            max_val = max(max_val, input[offset + i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += expf(input[offset + i] - max_val);
        }
        
        for (int i = 0; i < N; i++) {
            output[offset + i] = expf(input[offset + i] - max_val) / sum;
        }
    }
}

void softmax(float *x, int N) {
    float max = x[0];
    for (int i = 1; i < N; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < N; i++) {
        x[i] /= sum;
    }
}

int main() {
    const int B = 32;  // Batch size
    const int N = 1024;  // Row length
    float *x_cpu = (float*)malloc(B * N * sizeof(float));
    float *x_gpu = (float*)malloc(B * N * sizeof(float));
    float *d_input, *d_output;

    // Initialize input vector
    for (int i = 0; i < B * N; i++) {
        x_cpu[i] = (float)rand() / RAND_MAX;  // Random values between 0 and 1
        x_gpu[i] = x_cpu[i];  // Copy to GPU input
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, B * N * sizeof(float));
    cudaMalloc((void**)&d_output, B * N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, x_gpu, B * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid_x = (N + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(blocksPerGrid_x, B);
    softmax_cuda<<<gridDim, threadsPerBlock>>>(d_input, d_output, B, N);

    // Copy result back to host
    cudaMemcpy(x_gpu, d_output, B * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute softmax on CPU (for one batch as an example)
    softmax(x_cpu, N);

    // Compare results (for the first batch as an example)
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(x_cpu[i] - x_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("Maximum difference between CPU and GPU results (first batch): %e\n", max_diff);

    // Clean up
    free(x_cpu);
    free(x_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}