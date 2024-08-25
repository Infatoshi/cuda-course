#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(1); } }

// Naive 2D convolution kernel
__global__ void naiveConv2D(float* input, float* kernel, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = -3; ky <= 3; ky++) {
            for (int kx = -3; kx <= 3; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    sum += input[iy * width + ix] * kernel[(ky + 3) * 7 + (kx + 3)];
                }
            }
        }
        output[y * width + x] = sum;
    }
}

// Helper function for CUDA timing
float milliseconds(clock_t clock_value) {
    return (float)clock_value * 1000.0 / CLOCKS_PER_SEC;
}

int main() {
    const int width = 8192;
    const int height = 8192;
    const int kernelSize = 7;
    const int inputSize = width * height;
    const int outputSize = inputSize;

    std::cout << "Image size: " << width << "x" << height << std::endl;
    std::cout << "Kernel size: " << kernelSize << "x" << kernelSize << std::endl;
    // Allocate host memory
    float* h_input = (float*)malloc(inputSize * sizeof(float));
    float* h_kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    float* h_output_cudnn = (float*)malloc(outputSize * sizeof(float));
    float* h_output_naive = (float*)malloc(outputSize * sizeof(float));

    // Initialize input and kernel
    for (int i = 0; i < inputSize; i++) h_input[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < kernelSize * kernelSize; i++) h_kernel[i] = rand() / (float)RAND_MAX;

    // // print the first few elements of the input and kernel
    // printf("Input:\n");
    // for (int i = 0; i < 10; i++) {
    //   printf("%f ", h_input[i]);
    // }
    // printf("\n");
    // printf("Kernel:\n");
    // for (int i = 0; i < 10; i++) {
    //   printf("%f ", h_kernel[i]);
    // }
    // printf("\n");

    // Allocate device memory
    float *d_input, *d_kernel, *d_output_cudnn, *d_output_naive;
    CHECK_CUDA(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, outputSize * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));

    // cuDNN setup
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, kernelSize, kernelSize));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, 3, 3, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Request all available algorithms
    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, inputDesc, kernelDesc, convDesc, outputDesc,
                                                       requestedAlgoCount, &returnedAlgoCount, perfResults));

    // Find the fastest algorithm
    cudnnConvolutionFwdAlgo_t fastestAlgo = perfResults[0].algo;
    float fastestTime = perfResults[0].time;
    for (int i = 1; i < returnedAlgoCount; i++) {
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS && perfResults[i].time < fastestTime) {
            fastestAlgo = perfResults[i].algo;
            fastestTime = perfResults[i].time;
        }
    }

    cudnnConvolutionFwdAlgo_t algo = fastestAlgo;

    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize));

    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // Define grid and block sizes for the naive kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Warmup and benchmark runs
    const int warmupRuns = 3;
    const int benchmarkRuns = 100;
    float totalTime_cudnn = 0.0f;
    float totalTime_naive = 0.0f;

    float alpha = 1.0f, beta = 0.0f;

    // Warmup runs
    for (int i = 0; i < warmupRuns; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        naiveConv2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Benchmark runs
    for (int i = 0; i < benchmarkRuns; i++) {
        // cuDNN benchmark
        cudaEvent_t start_cudnn, stop_cudnn;
        CHECK_CUDA(cudaEventCreate(&start_cudnn));
        CHECK_CUDA(cudaEventCreate(&stop_cudnn));
        
        CHECK_CUDA(cudaEventRecord(start_cudnn));
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        CHECK_CUDA(cudaEventRecord(stop_cudnn));
        CHECK_CUDA(cudaEventSynchronize(stop_cudnn));
        
        float milliseconds_cudnn = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds_cudnn, start_cudnn, stop_cudnn));
        totalTime_cudnn += milliseconds_cudnn;

        CHECK_CUDA(cudaEventDestroy(start_cudnn));
        CHECK_CUDA(cudaEventDestroy(stop_cudnn));

        // Naive kernel benchmark
        cudaEvent_t start_naive, stop_naive;
        CHECK_CUDA(cudaEventCreate(&start_naive));
        CHECK_CUDA(cudaEventCreate(&stop_naive));
        
        CHECK_CUDA(cudaEventRecord(start_naive));
        naiveConv2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height);
        CHECK_CUDA(cudaEventRecord(stop_naive));
        CHECK_CUDA(cudaEventSynchronize(stop_naive));
        
        float milliseconds_naive = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds_naive, start_naive, stop_naive));
        totalTime_naive += milliseconds_naive;

        CHECK_CUDA(cudaEventDestroy(start_naive));
        CHECK_CUDA(cudaEventDestroy(stop_naive));
    }

    // Calculate average times
    float avgTime_cudnn = totalTime_cudnn / benchmarkRuns;
    float avgTime_naive = totalTime_naive / benchmarkRuns;

    printf("cuDNN average time: %f ms\n", avgTime_cudnn);
    printf("Naive kernel average time: %f ms\n", avgTime_naive);

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    float maxDiff = 0.0f;
    for (int i = 0; i < outputSize; i++) {
        float diff = fabs(h_output_cudnn[i] - h_output_naive[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    printf("Max difference between cuDNN and naive kernel: %f\n", maxDiff);

    // Clean up
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernelDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output_cudnn));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_workspace));

    free(h_input);
    free(h_kernel);
    free(h_output_cudnn);
    free(h_output_naive);

    return 0;
}