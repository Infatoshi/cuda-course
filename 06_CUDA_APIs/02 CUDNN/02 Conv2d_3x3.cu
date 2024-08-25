#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(1); } }

// Naive 2D convolution kernel with channels
__global__ void naiveConv2D(float* input, float* kernel, float* output, int width, int height, int inChannels, int outChannels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outChannel = blockIdx.z;

    if (x < width && y < height && outChannel < outChannels) {
        float sum = 0.0f;
        for (int inChannel = 0; inChannel < inChannels; inChannel++) {
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int inputIdx = (inChannel * height * width) + (iy * width + ix);
                        int kernelIdx = (outChannel * inChannels * 9) + (inChannel * 9) + ((ky + 1) * 3 + (kx + 1));
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }
        output[(outChannel * height * width) + (y * width + x)] = sum;
    }
}

// Helper function for CUDA timing
float milliseconds(clock_t clock_value) {
    return (float)clock_value * 1000.0 / CLOCKS_PER_SEC;
}

// CPU 2D convolution function with channels
void cpuConv2D(float* input, float* kernel, float* output, int width, int height, int inChannels, int outChannels) {
    for (int outChannel = 0; outChannel < outChannels; outChannel++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum = 0.0f;
                for (int inChannel = 0; inChannel < inChannels; inChannel++) {
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int ix = x + kx;
                            int iy = y + ky;
                            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                                int inputIdx = (inChannel * height * width) + (iy * width + ix);
                                int kernelIdx = (outChannel * inChannels * 9) + (inChannel * 9) + ((ky + 1) * 3 + (kx + 1));
                                sum += input[inputIdx] * kernel[kernelIdx];
                            }
                        }
                    }
                }
                output[(outChannel * height * width) + (y * width + x)] = sum;
            }
        }
    }
}

int main() {
    const int width = 256;
    const int height = 256;
    const int kernelSize = 3;
    const int inChannels = 3;
    const int outChannels = 3;
    const int inputSize = width * height * inChannels;
    const int outputSize = width * height * outChannels;
    const int kernelTotalSize = kernelSize * kernelSize * inChannels * outChannels;

    std::cout << "Image size: " << width << "x" << height << "x" << inChannels << std::endl;
    std::cout << "Kernel size: " << kernelSize << "x" << kernelSize << "x" << inChannels << "x" << outChannels << std::endl;
    // Allocate host memory
    float* h_input = (float*)malloc(inputSize * sizeof(float));
    float* h_kernel = (float*)malloc(kernelTotalSize * sizeof(float));
    float* h_output_cudnn = (float*)malloc(outputSize * sizeof(float));
    float* h_output_naive = (float*)malloc(outputSize * sizeof(float));
    float* h_output_cpu = (float*)malloc(outputSize * sizeof(float));

    // Initialize input and kernel
    for (int i = 0; i < inputSize; i++) h_input[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < kernelTotalSize; i++) h_kernel[i] = rand() / (float)RAND_MAX;

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
    CHECK_CUDA(cudaMalloc(&d_kernel, kernelTotalSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, outputSize * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernelTotalSize * sizeof(float), cudaMemcpyHostToDevice));

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

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inChannels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outChannels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, inChannels, kernelSize, kernelSize));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, outChannels);

    // Warmup and benchmark runs
    const int warmupRuns = 3;
    const int benchmarkRuns = 100;
    float totalTime_cudnn = 0.0f;
    float totalTime_naive = 0.0f;
    float totalTime_cpu = 0.0f;

    float alpha = 1.0f, beta = 0.0f;

    // Warmup runs
    for (int i = 0; i < warmupRuns; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        naiveConv2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height, inChannels, outChannels);
        cpuConv2D(h_input, h_kernel, h_output_cpu, width, height, inChannels, outChannels);
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
        naiveConv2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height, inChannels, outChannels);
        CHECK_CUDA(cudaEventRecord(stop_naive));
        CHECK_CUDA(cudaEventSynchronize(stop_naive));
        
        float milliseconds_naive = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds_naive, start_naive, stop_naive));
        totalTime_naive += milliseconds_naive;

        CHECK_CUDA(cudaEventDestroy(start_naive));
        CHECK_CUDA(cudaEventDestroy(stop_naive));

        // CPU benchmark
        clock_t start_cpu = clock();
        cpuConv2D(h_input, h_kernel, h_output_cpu, width, height, inChannels, outChannels);
        clock_t end_cpu = clock();
        totalTime_cpu += milliseconds(end_cpu - start_cpu);
    }

    // Calculate average times
    float avgTime_cudnn = totalTime_cudnn / benchmarkRuns;
    float avgTime_naive = totalTime_naive / benchmarkRuns;
    float avgTime_cpu = totalTime_cpu / benchmarkRuns;

    printf("cuDNN average time: %f ms\n", avgTime_cudnn);
    printf("Naive kernel average time: %f ms\n", avgTime_naive);
    printf("CPU average time: %f ms\n", avgTime_cpu);

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    float maxDiff_cudnn_cpu = 0.0f;
    float maxDiff_naive_cpu = 0.0f;
    for (int i = 0; i < outputSize; i++) {
        float diff_cudnn = fabs(h_output_cudnn[i] - h_output_cpu[i]);
        float diff_naive = fabs(h_output_naive[i] - h_output_cpu[i]);
        if (diff_cudnn > maxDiff_cudnn_cpu) maxDiff_cudnn_cpu = diff_cudnn;
        if (diff_naive > maxDiff_naive_cpu) maxDiff_naive_cpu = diff_naive;
    }

    printf("Max difference between cuDNN and CPU: %f\n", maxDiff_cudnn_cpu);
    printf("Max difference between naive and CPU: %f\n", maxDiff_naive_cpu);

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
    free(h_output_cpu);

    return 0;
}