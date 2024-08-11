#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void polynomial_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    size_t size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        const scalar_t x = input[index];
        output[index] = x * x + x + 1;
    }
}

template <typename scalar_t>
__global__ void polynomial_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ grad_input,
    size_t size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        grad_input[index] = grad_output[index] * (2 * input[index] + 1);
    }
}

torch::Tensor polynomial_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 1024;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "polynomial_forward_cuda", ([&] {
        polynomial_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

torch::Tensor polynomial_cuda_backward(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::empty_like(input);
    const int threads = 1024;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "polynomial_backward_cuda", ([&] {
        polynomial_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return grad_input;
}