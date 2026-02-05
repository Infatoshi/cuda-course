# cuBLAS-Lt

- when I was initally testing this, I was getting cublas errors because I was making the matrices smaller (so I could understand where incorrect results were coming from)
- from [cublas-lt](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul) docs, search for "Dimensions m and k must be multiples of 4."
- this means we can't have a 3x4 or 2x4 matrix, but we can have a 4x4 or 4x8 matrix

## Compilation Instructions
- Compile the code using the `nvcc` compiler with the following command:
  ```sh
  nvcc -o matmul main.cu -lcublasLt -lcublas -lcuda
  ```
- Run the compiled executable:
  ```sh
   ./matmul
  ```
  where main.cu is the name of the file
