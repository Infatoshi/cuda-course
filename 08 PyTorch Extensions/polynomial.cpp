#include <torch/extension.h>

torch::Tensor polynomial_forward(torch::Tensor input) {
    return input * input + input + 1;
}

torch::Tensor polynomial_backward(torch::Tensor grad_output, torch::Tensor input) {
    return grad_output * (2 * input + 1);
}

// Declare the CUDA functions
torch::Tensor polynomial_cuda_forward(torch::Tensor input);
torch::Tensor polynomial_cuda_backward(torch::Tensor grad_output, torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &polynomial_forward, "Polynomial forward (CPU)");
    m.def("backward", &polynomial_backward, "Polynomial backward (CPU)");
    m.def("forward_cuda", &polynomial_cuda_forward, "Polynomial forward (CUDA)");
    m.def("backward_cuda", &polynomial_cuda_backward, "Polynomial backward (CUDA)");
}