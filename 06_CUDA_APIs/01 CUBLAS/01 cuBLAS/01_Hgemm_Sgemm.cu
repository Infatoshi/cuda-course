// dedicated for small handwritten matrices
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define M 3
#define K 4
#define N 2

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols) \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) \
            printf("%8.3f ", mat[i * cols + j]); \
        printf("\n"); \
    } \
    printf("\n");

void cpu_matmul(float *A, float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float B[K * N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float C_cpu[M * N], C_cublas_s[M * N], C_cublas_h[M * N];

    // CPU matmul
    cpu_matmul(A, B, C_cpu);

    // CUDA setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // row major A = 
    // 1.0 2.0 3.0 4.0
    // 5.0 6.0 7.0 8.0

    // col major A = 
    // 1.0 5.0
    // 2.0 6.0
    // 3.0 7.0
    // 4.0 8.0

    // memory layout (row)
    // 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0

    // memory layout (col)
    // 1.0 5.0 2.0 6.0 3.0 7.0 4.0 8.0
    
    // cuBLAS SGEMM
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUDA(cudaMemcpy(C_cublas_s, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // cuBLAS HGEMM
    half *d_A_h, *d_B_h, *d_C_h;
    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

    // Convert to half precision on CPU
    half A_h[M * K], B_h[K * N];
    for (int i = 0; i < M * K; i++) {
        A_h[i] = __float2half(A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        B_h[i] = __float2half(B[i]);
    }

    // Copy half precision data to device
    CHECK_CUDA(cudaMemcpy(d_A_h, A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));

    // Copy result back to host and convert to float
    half C_h[M * N];
    CHECK_CUDA(cudaMemcpy(C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++) {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    // Print results
    printf("Matrix A (%dx%d):\n", M, K);
    PRINT_MATRIX(A, M, K);
    printf("Matrix B (%dx%d):\n", K, N);
    PRINT_MATRIX(B, K, N);
    printf("CPU Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cpu, M, N);
    printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_s, M, N);
    printf("cuBLAS HGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_h, M, N);

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h));
    CHECK_CUDA(cudaFree(d_B_h));
    CHECK_CUDA(cudaFree(d_C_h));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}