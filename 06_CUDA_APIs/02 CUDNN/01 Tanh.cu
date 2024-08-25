#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(1); } }

const int N = 1 << 24;  // Number of elements 16M
const int WARMUP_RUNS = 10;
const int BENCHMARK_RUNS = 100;

__global__ void naiveTanhKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

void benchmarkNaiveTanh(float* d_input, float* d_output) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        naiveTanhKernel<<<grid, block>>>(d_input, d_output, N);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        naiveTanhKernel<<<grid, block>>>(d_input, d_output, N);
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
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

    const float alpha = 1.0f, beta = 0.0f;

    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        CHECK_CUDNN(cudnnActivationForward(cudnn, activationDesc, &alpha, inputDesc, d_input, &beta, outputDesc, d_output));
    }
    cudaDeviceSynchronize();

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

int main() {
    float *h_input, *d_input, *d_output;
    h_input = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 2 - 1;  // Random values between -1 and 1
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Benchmark naive CUDA implementation
    benchmarkNaiveTanh(d_input, d_output);

    // Benchmark cuDNN implementation
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    benchmarkCuDNNTanh(cudnn, d_input, d_output);

    // Cleanup
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}