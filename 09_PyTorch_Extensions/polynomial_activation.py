import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.utils.cpp_extension import load
import time

# PyTorch built-in implementation
class PolynomialActivation(nn.Module):
    def forward(self, x):
        return x**2 + x + 1

# Load CUDA extension
cuda_extension = load(
    name="polynomial_cuda",
    sources=["polynomial_cuda.cpp", "polynomial_cuda_kernel.cu"],
    verbose=True
)

# Triton implementation
@triton.jit
def polynomial_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * x + x + 1
    tl.store(output_ptr + offsets, output, mask=mask)

class TritonPolynomialActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = torch.empty_like(x)
        grid = (triton.cdiv(x.numel(), 1024),)
        polynomial_kernel[grid](x, output, x.numel(), 1024)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")

# Benchmark function
def benchmark(func, x, name, num_runs=1000):
    start_time = time.time()
    for _ in range(num_runs):
        func(x)
    torch.cuda.synchronize()
    end_time = time.time()
    return f"{name}: {(end_time - start_time) / num_runs * 1000:.4f} ms"

# Main function to run benchmarks
def main():
    torch.manual_seed(0)
    x = torch.randn(1000000, device='cuda')

    # PyTorch built-in
    pytorch_activation = PolynomialActivation().cuda()
    pytorch_time = benchmark(pytorch_activation, x, "PyTorch built-in")

    # CUDA extension
    cuda_time = benchmark(cuda_extension.polynomial_activation, x, "CUDA extension")

    # Triton
    triton_time = benchmark(TritonPolynomialActivation.apply, x, "Triton")

    print(pytorch_time)
    print(cuda_time)
    print(triton_time)

if __name__ == "__main__":
    main()