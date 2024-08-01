import torch
from torch.autograd import Function
import polynomial_cpp

class PolynomialFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            return polynomial_cpp.forward_cuda(input)
        else:
            return polynomial_cpp.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if input.is_cuda:
            return polynomial_cpp.backward_cuda(grad_output, input)
        else:
            return polynomial_cpp.backward(grad_output, input)

class Polynomial(torch.nn.Module):
    def forward(self, input):
        return PolynomialFunction.apply(input)

# Test the implementation
if __name__ == "__main__":
    # Test on CPU
    x_cpu = torch.randn(1000000, requires_grad=True)
    poly_cpu = Polynomial()
    
    # Test on GPU if available
    if torch.cuda.is_available():
        x_gpu = torch.randn(1000000, requires_grad=True, device='cuda')
        poly_gpu = Polynomial()

    # CPU forward and backward
    y_cpu = poly_cpu(x_cpu)
    loss_cpu = y_cpu.sum()
    loss_cpu.backward()

    # GPU forward and backward
    if torch.cuda.is_available():
        y_gpu = poly_gpu(x_gpu)
        loss_gpu = y_gpu.sum()
        loss_gpu.backward()

    # Print results
    print("CPU result:", y_cpu[:5])
    print("CPU gradient:", x_cpu.grad[:5])
    
    if torch.cuda.is_available():
        print("GPU result:", y_gpu[:5])
        print("GPU gradient:", x_gpu.grad[:5])

    # Verify results
    expected_y = x_cpu**2 + x_cpu + 1
    expected_grad = 2*x_cpu + 1
    print("\nCPU forward pass difference:", torch.max(torch.abs(y_cpu - expected_y)))
    print("CPU backward pass difference:", torch.max(torch.abs(x_cpu.grad - expected_grad)))

    if torch.cuda.is_available():
        print("GPU forward pass difference:", torch.max(torch.abs(y_gpu.cpu() - expected_y)))
        print("GPU backward pass difference:", torch.max(torch.abs(x_gpu.grad.cpu() - expected_grad)))