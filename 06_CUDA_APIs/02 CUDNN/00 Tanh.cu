#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudnnGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Naive CUDA kernel for tanh activation
__global__ void naiveTanhKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// Utility function for CPU verification
float cpuTanh(float x) {
    return tanhf(x);
}

// Function to initialize data
void initializeData(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random values between -1 and 1
    }
}

// Function to verify results
bool verifyResults(float* cpu_output, float* gpu_output, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (fabs(cpu_output[i] - gpu_output[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_output[i], gpu_output[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Set up tensor dimensions for a scenario where cuDNN is likely to outperform
    const int batch_size = 256; // NCHW format
    const int channels = 32;
    const int height = 224;
    const int width = 224;
    const int tensor_size = batch_size * channels * height * width;

    // Allocate host memory
    float *h_input, *h_output_naive, *h_output_cudnn, *h_output_cpu;
    h_input = (float*)malloc(tensor_size * sizeof(float));
    h_output_naive = (float*)malloc(tensor_size * sizeof(float));
    h_output_cudnn = (float*)malloc(tensor_size * sizeof(float));
    h_output_cpu = (float*)malloc(tensor_size * sizeof(float));

    // Initialize input data
    initializeData(h_input, tensor_size);

    // Allocate device memory
    float *d_input, *d_output_naive, *d_output_cudnn;
    CHECK_CUDA(cudaMalloc(&d_input, tensor_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, tensor_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, tensor_size * sizeof(float)));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, tensor_size * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup and benchmark parameters
    const int num_warmup = 10;
    const int num_benchmark = 100;
    float naive_times[num_benchmark];
    float cudnn_times[num_benchmark];

    // Naive CUDA kernel
    dim3 block(256);
    dim3 grid((tensor_size + block.x - 1) / block.x);

    // Warmup runs for naive kernel
    for (int i = 0; i < num_warmup; ++i) {
        naiveTanhKernel<<<grid, block>>>(d_input, d_output_naive, tensor_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs for naive kernel
    for (int i = 0; i < num_benchmark; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        naiveTanhKernel<<<grid, block>>>(d_input, d_output_naive, tensor_size);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&naive_times[i], start, stop));
    }

    // cuDNN setup
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, channels, height, width));

    cudnnActivationDescriptor_t activation_descriptor;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_TANH,
                                             CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;

    // Warmup runs for cuDNN
    for (int i = 0; i < num_warmup; ++i) {
        CHECK_CUDNN(cudnnActivationForward(cudnn, activation_descriptor, &alpha, input_descriptor, d_input,
                                           &beta, input_descriptor, d_output_cudnn));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs for cuDNN
    for (int i = 0; i < num_benchmark; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnActivationForward(cudnn, activation_descriptor, &alpha, input_descriptor, d_input,
                                           &beta, input_descriptor, d_output_cudnn));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&cudnn_times[i], start, stop));
    }

    // Calculate average times
    float avg_naive_time = 0.0f, avg_cudnn_time = 0.0f;
    for (int i = 0; i < num_benchmark; ++i) {
        avg_naive_time += naive_times[i];
        avg_cudnn_time += cudnn_times[i];
    }
    avg_naive_time /= num_benchmark;
    avg_cudnn_time /= num_benchmark;

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, tensor_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn, tensor_size * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU verification
    for (int i = 0; i < tensor_size; ++i) {
        h_output_cpu[i] = cpuTanh(h_input[i]);
    }

    // Verify results
    bool naive_correct = verifyResults(h_output_cpu, h_output_naive, tensor_size);
    bool cudnn_correct = verifyResults(h_output_cpu, h_output_cudnn, tensor_size);

    // Print results
    printf("Tensor size: %d x %d x %d x %d\n", batch_size, channels, height, width);
    printf("Average Naive CUDA kernel time: %.3f ms\n", avg_naive_time);
    printf("Average cuDNN activation time: %.3f ms\n", avg_cudnn_time);
    printf("Speedup: %.2fx\n", avg_naive_time / avg_cudnn_time);
    printf("Naive kernel results correct: %s\n", naive_correct ? "Yes" : "No");
    printf("cuDNN results correct: %s\n", cudnn_correct ? "Yes" : "No");

    // Clean up
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_output_cudnn));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));
    CHECK_CUDNN(cudnnDestroy(cudnn));
    free(h_input);
    free(h_output_naive);
    free(h_output_cudnn);
    free(h_output_cpu);

    return 0;
}