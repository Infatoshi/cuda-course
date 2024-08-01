



## Purpose of `pragma #unroll`
- ideally, you'd want more useful compute per iteration. if you can do 4 math operations inside of 1 per iteration thats good.
- in some contexts, the compiler will actually will actually unroll the loop without explicitly telling it to do so. (this is what happened with `unrolling.cu`)
- you can check the PTX assembly code with `nvcc -ptx v1.cu -o - | less` to see if the compiler has unrolled the loop.
- by writing a kernel without unrolling and benchmarking it with a kernel that has unrolling, you can see if the unrolling 
  is actually beneficial. then check the PTX assembly code to see if the compiler has unrolled the loop. only beneficial if you aren't getting the benefits you wanted and need to investigate further.
- the quickly benchmark, just take the average time of the kernel and compare it to the unrolled version. if the unrolled version is faster, then the unrolling was beneficial. if not, then the unrolling was not beneficial. always make sure to verify results so your kernel is outputting what is should (compare element-wise)


## What is occupancy
    
    Occupancy is defined as the ratio between the number of active warps per SM and the maximum possible number of active warps per SM.
    
    There are three main limits to keeping more active blocks loaded on an SM: register count, warp count and SMEM capacity. Let’s do an example calculation for our current kernel.
    
    https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy

> [Matmul Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)