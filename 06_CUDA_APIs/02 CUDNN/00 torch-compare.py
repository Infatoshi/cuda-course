import torch
import time
import math

# Constants
N = 1 << 19  # Number of elements (8192)
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
BATCH_SIZE = 256

# Custom tanh implementation
def custom_tanh(x):
    return (torch.exp(2*x) - 1) / (torch.exp(2*x) + 1)

def benchmark_custom_tanh(input_tensor):
    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = custom_tanh(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        _ = custom_tanh(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) * 1000 / BENCHMARK_RUNS  # Convert to milliseconds
    print(f"Custom Tanh: Avg time per run: {avg_time:.3f} ms")
    
    return custom_tanh(input_tensor)

def benchmark_builtin_tanh(input_tensor):
    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = torch.tanh(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        _ = torch.tanh(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) * 1000 / BENCHMARK_RUNS  # Convert to milliseconds
    print(f"Built-in Tanh: Avg time per run: {avg_time:.3f} ms")
    
    return torch.tanh(input_tensor)

def verify_outputs(custom_output, builtin_output):
    max_diff = torch.max(torch.abs(custom_output - builtin_output)).item()
    print(f"Max difference between custom and built-in outputs: {max_diff:.6e}")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate input data
    input_tensor = torch.rand((128, 32, 224, 224), device=device) * 2 - 1  # Random values between -1 and 1

    # Warm up GPU
    _ = torch.tanh(input_tensor)
    torch.cuda.synchronize()

    # Benchmark custom tanh
    custom_output = benchmark_custom_tanh(input_tensor)

    # Benchmark built-in tanh
    builtin_output = benchmark_builtin_tanh(input_tensor)

    # Verify outputs
    verify_outputs(custom_output, builtin_output)

if __name__ == "__main__":
    main()