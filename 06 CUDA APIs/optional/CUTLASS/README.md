# CUTLASS

- [CUDA Templates for Linear Algebra Subroutines and Solvers ⇒ cutlass & ‣](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

- Most commonly used for Matrix multiplication since its the heart of the transformer architecture so we care about `cutlass/gemm`
- you likely won’t ever use cutlass just because of the overarching complexity. cutlass is designed for kernel engineers to fine-tune and optimize around specific hardware architectures whilst requiring deep GPU kernel knowledge.
- in the comparison script below, I compare the time taken for matrix multiplication on cuBLAS vs CUTLASS. we use 1024x1024x1024 matrices, warmup cuBLAS with 10 matmuls then record time, then do the same for CUTLASS.
- This is performance difference (not super massive but we will happily take the ~10% boost w/ cuBLAS)
    
```bash
cuBLAS Time: 0.202861 ms
CUTLASS Time: 0.227451 ms
```

- you have you pass in the path for cutlass during compilation with `-I` flag. consider writing a seperate path for cutlass in your `~/.bashrc` or `~/.zshrc` file for ease of use


