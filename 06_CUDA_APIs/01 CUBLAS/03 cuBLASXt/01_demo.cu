#include <cublasXt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Define matrix dimensions
const int M = 1024 / 4;
const int N = 1024 / 4;
const int K = 1024 / 4;

#define CHECK_CUBLAS(call) { cublasStatus_t err = call; if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "Error in " << #call << ", line " << __LINE__ << std::endl; exit(1); } }

int main() {
    // Initialize random number generator
    srand(time(0));

    // Allocate host memory for matrices
    float* A_host = new float[M * K];
    float* B_host = new float[K * N];
    float* C_host_cpu = new float[M * N];
    float* C_host_gpu = new float[M * N];

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) {
        A_host[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B_host[i] = (float)rand() / RAND_MAX;
    }

    // Perform CPU-based matrix multiplication
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_host_cpu[i * N + j] = 0.0f;
            for (int k = 0; k < K; k++) {
                C_host_cpu[i * N + j] += A_host[i * K + k] * B_host[k * N + j];
            }
        }
    }
    
    cublasXtHandle_t handle;
    CHECK_CUBLAS(cublasXtCreate(&handle));

    int devices[1] = {0};
    CHECK_CUBLAS(cublasXtDeviceSelect(handle, 1, devices));

    // Warmup run
    CHECK_CUBLAS(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_host, N, A_host, K, &beta, C_host_gpu, N));


    // Compare results
    float max_diff = 1e-4f;
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(C_host_cpu[i] - C_host_gpu[i]);
        if (diff > max_diff) {
            std::cout << "i: " << i << " CPU: " << C_host_cpu[i] << ", GPU: " << C_host_gpu[i] << std::endl;
            
        }
    }
    std::cout << "Maximum difference between CPU and GPU results: " << max_diff << std::endl;

    // Free memory
    delete[] A_host;
    delete[] B_host;
    delete[] C_host_cpu;
    delete[] C_host_gpu;


    return 0;
}