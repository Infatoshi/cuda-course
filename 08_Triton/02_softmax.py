import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    row_idx = tl.program_id(axis=0)

    # Compute the memory offsets for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Load the row into SRAM
    row = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))

    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max from row and exponentiate
    numerator = tl.exp(row - row_max)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store the output
    tl.store(out_row_start_ptr + tl.arange(0, BLOCK_SIZE), softmax_output, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  
    
    # Launch the Triton kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return output

# Set up the input tensor
torch.manual_seed(0)
x = torch.randn(256, 1024, device='cuda')
# x = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
# Compute softmax using PyTorch
torch_result = torch.softmax(x, dim=1)

# Compute softmax using Triton
triton_result = triton_softmax(x)

# Compare results
max_diff = torch.max(torch.abs(torch_result - triton_result))
print(f"Maximum difference between PyTorch and Triton results: {max_diff:.2e}")

# Check if results are close
is_close = torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)
print(f"Results are close: {is_close}")