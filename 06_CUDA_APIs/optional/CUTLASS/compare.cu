#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/tensor_view_io.h>

#define CHECK_CUDA(call) \
    if((call) != cudaSuccess) { \
        std::cerr << "CUDA error at: " << __FILE__ << ":" << __LINE__ << " : " << cudaGetErrorString(call) << std::endl; \
        exit(1); \
    }

#define CHECK_CUBLAS(call) \
    if((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at: " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

void verify_results(const std::vector<float>& A, const std::vector<float>& B, int rows, int cols) {
    float epsilon = 1e-2; // Adjust if necessary
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(A[i] - B[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": A[i] = " << A[i] << ", B[i] = " << B[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "Outputs match within tolerance." << std::endl;
}

int main() {
    int m = 1024, n = 1024, k = 1024;
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C_cublas(m * n, 0.0f);
    std::vector<float> h_C_cutlass(m * n, 0.0f);

    // Initialize matrices with random values
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < m * k; ++i) h_A[i] = dist(rng);
    for (int i = 0; i < k * n; ++i) h_B[i] = dist(rng);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warmup cuBLAS
    for (int i = 0; i < 10; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cublas_time = end - start;
    std::cout << "cuBLAS Time: " << cublas_time.count() << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // CUTLASS
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor, float, ColumnMajor>;

    Gemm gemm_op;
    Gemm::Arguments args({m, n, k}, {d_A, m}, {d_B, k}, {d_C, m}, {d_C, m}, {alpha, beta});
    
    // Warmup CUTLASS
    for (int i = 0; i < 10; ++i) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed: " << cutlassGetStatusString(status) << std::endl;
            exit(1);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    start = std::chrono::high_resolution_clock::now();
    cutlass::Status status = gemm_op(args);
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cutlass_time = end - start;

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed: " << cutlassGetStatusString(status) << std::endl;
        exit(1);
    }

    std::cout << "CUTLASS Time: " << cutlass_time.count() << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C_cutlass.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Verify results
    verify_results(h_C_cublas, h_C_cutlass, m, n);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}