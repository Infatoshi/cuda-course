#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(1); } }

const int N = 1 << 19;  // Number of elements 8192
const int WARMUP_RUNS = 10;
const int BENCHMARK_RUNS = 100;
const int BATCH_SIZE = 256;  // New variable for batch size

__global__ void naiveTanhKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // torch.tanh
        output[idx] = tanhf(input[idx]);
        // (torch.exp(2*x) - 1) / (torch.exp(2*x) + 1)
        // output[idx] = (expf(2 * input[idx]) - 1) / (expf(2 * input[idx]) + 1);
    }
}

void benchmarkNaiveTanh(float* d_input, float* d_output) {
    dim3 block(256);
    dim3 grid((N * BATCH_SIZE + block.x - 1) / block.x);

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        naiveTanhKernel<<<grid, block>>>(d_input, d_output, N * BATCH_SIZE);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        naiveTanhKernel<<<grid, block>>>(d_input, d_output, N * BATCH_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive Tanh: Avg time per run: %.3f ms\n", milliseconds / BENCHMARK_RUNS);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmarkCuDNNTanh(cudnnHandle_t cudnn, float* d_input, float* d_output) {
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, 1, 1, N));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, 1, 1, N));

    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));

    const float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        CHECK_CUDNN(cudnnActivationForward(cudnn, activationDesc, &alpha, inputDesc, d_input, &beta, outputDesc, d_output));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        CHECK_CUDNN(cudnnActivationForward(cudnn, activationDesc, &alpha, inputDesc, d_input, &beta, outputDesc, d_output));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuDNN Tanh: Avg time per run: %.3f ms\n", milliseconds / BENCHMARK_RUNS);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyActivationDescriptor(activationDesc);
}

void warmupGPU(float* d_input, float* d_output) {
    // Perform a dummy kernel launch to warm up the GPU
    dim3 block(256);
    dim3 grid((N * BATCH_SIZE + block.x - 1) / block.x);
    naiveTanhKernel<<<grid, block>>>(d_input, d_output, N * BATCH_SIZE);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void verifyOutputs(float* naive_output, float* cudnn_output) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * BATCH_SIZE; i++) {
        float diff = fabsf(naive_output[i] - cudnn_output[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max difference between naive and cuDNN outputs: %e\n", max_diff);
}

int main() {
    float *h_input, *d_input, *d_output;
    h_input = (float*)malloc(N * BATCH_SIZE * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_input, N * BATCH_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * BATCH_SIZE * sizeof(float)));

    // Initialize input data
    for (int i = 0; i < N * BATCH_SIZE; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 2 - 1;  // Random values between -1 and 1
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    warmupGPU(d_input, d_output);

    float *h_naive_output = (float*)malloc(N * BATCH_SIZE * sizeof(float));
    float *h_cudnn_output = (float*)malloc(N * BATCH_SIZE * sizeof(float));

    // Benchmark naive CUDA implementation
    benchmarkNaiveTanh(d_input, d_output);
    CHECK_CUDA(cudaMemcpy(h_naive_output, d_output, N * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Benchmark cuDNN implementation
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    benchmarkCuDNNTanh(cudnn, d_input, d_output);
    CHECK_CUDA(cudaMemcpy(h_cudnn_output, d_output, N * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    verifyOutputs(h_naive_output, h_cudnn_output);

    // Cleanup
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    free(h_naive_output);
    free(h_cudnn_output);

    return 0;
}