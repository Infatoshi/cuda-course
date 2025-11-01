#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000000
#define THREADS_PER_BLOCK 256
#define LOOP_COUNT 100
#define WARMUP_RUNS 5
#define BENCH_RUNS 10

// Kernel without loop unrolling
__global__ void vectorAddNoUnroll(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float sum = 0;
        for (int j = 0; j < LOOP_COUNT; j++) {
            sum += a[tid] + b[tid];
        }
        c[tid] = sum;
    }
}

// Kernel with loop unrolling using #pragma unroll
__global__ void vectorAddUnroll(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float sum = 0;
        #pragma unroll
        for (int j = 0; j < LOOP_COUNT; j++) {
            sum += a[tid] + b[tid];
        }
        c[tid] = sum;
    }
}

// Function to verify results
bool verifyResults(float *c, int n) {
    float expected = (1.0f + 2.0f) * LOOP_COUNT;
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - expected) > 1e-5) {
            return false;
        }
    }
    return true;
}

// Function to run kernel and measure time
float runKernel(void (*kernel)(float*, float*, float*, int), float *d_a, float *d_b, float *d_c, int n) {
    int numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t start, stop;
    float milliseconds;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy input data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Warmup runs
    for (int i = 0; i < WARMUP_RUNS; i++) {
        runKernel(vectorAddNoUnroll, d_a, d_b, d_c, N);
        runKernel(vectorAddUnroll, d_a, d_b, d_c, N);
    }

    // Benchmark runs
    float totalTimeNoUnroll = 0, totalTimeUnroll = 0;
    for (int i = 0; i < BENCH_RUNS; i++) {
        totalTimeNoUnroll += runKernel(vectorAddNoUnroll, d_a, d_b, d_c, N);
        totalTimeUnroll += runKernel(vectorAddUnroll, d_a, d_b, d_c, N);
    }

    // Calculate average times
    float avgTimeNoUnroll = totalTimeNoUnroll / BENCH_RUNS;
    float avgTimeUnroll = totalTimeUnroll / BENCH_RUNS;

    printf("Average time for kernel without unrolling: %f ms\n", avgTimeNoUnroll);
    printf("Average time for kernel with unrolling: %f ms\n", avgTimeUnroll);

    // Verify results
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    if (verifyResults(c, N)) {
        printf("Results are correct\n");
    } else {
        printf("Results are incorrect\n");
    }

    // Clean up
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}