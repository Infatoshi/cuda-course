# cuBLAS vs cuBLASLt Benchmark

## Environment and Matrix Size
- Matrix size: 4096x1024 * 1024x4096
- Code: 02_compare.cu

## Performance Results (Average Time)
- cuBLAS FP32: 3.33952 ms
- cuBLASLt FP32: 2.90258 ms
- cuBLAS FP16: 0.163584 ms
- cuBLASLt FP16: 0.162099 ms
- Naive CUDA kernel: 9.77638 ms

## Correctness and Error
- FP32 results match the naive kernel within tolerance 1e-2
- FP16 results match the naive kernel within tolerance 5e-1
- Max FP16 error: 0.0677938 (same for cuBLAS and cuBLASLt)

## Summary
- cuBLASLt is slightly faster than cuBLAS for FP32.
- FP16 performance is similar between the two and much faster than the naive kernel.
