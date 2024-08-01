#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

int main() {
    float *d_input, *d_kernel, *d_output_custom, *d_output_cudnn;
    cudaMalloc(&d_input, B*C*H*W*sizeof(float));
    cudaMalloc(&d_kernel, K*C*KH*KW*sizeof(float));
    cudaMalloc(&d_output_custom, B*K*H*W*sizeof(float));
    cudaMalloc(&d_output_cudnn, B*K*H*W*sizeof(float));

    // Copy input and kernel to device
    cudaMemcpy(d_input, h_input, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, K*C*KH*KW*sizeof(float), cudaMemcpyHostToDevice);

    

    // Kernel launch configuration
    dim3 blockDim(H, W);
    dim3 gridDim(B, K);
    custom_conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_output_custom, d_kernel, B, C, H, W, K, KH, KW);

    // Copy custom kernel output back to host
    cudaMemcpy(h_output_custom, d_output_custom, B*K*H*W*sizeof(float), cudaMemcpyDeviceToHost);

    // cuDNN setup
    cudnnHandle_t cudnn;
		  cudnnCreate(&cudnn);
		
		cudnnTensorDescriptor_t inputDesc, outputDesc;
		cudnnFilterDescriptor_t filterDesc;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, KH, KW);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    int n, c, h, w;
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &n, &c, &h, &w);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &n, &c, &h, &w);

    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize);

    void* workspace;
    cudaMalloc(&workspace, workspaceSize);

    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_kernel, convDesc, algo, workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));

}