# Kernels

## Kernel Launch Params

- Type `dim3` is 3D type for grids and thread blocks which are later feed into the kernel launch configuration.

- allows for indexing of elements as vector, matrix, or volume (tensor)

```cpp
dim3 gridDim(4, 4, 1); // 4 blocks in x, 4 block in y, 1 block in z
dim3 blockDim(4, 2, 2); // 4 threads in x, 2 thread in y, 2 thread in z
```

- other type is `int` which specifies a 1D vector

```cpp
int gridDim = 16; // 16 blocks
int blockDim = 32; // 32 threads per block
<<<gridDim, blockDim>>>
// these aren't dim3 types but they are still valid if the indexing scheme is 1D
```

- gridDim ⇒ gridDim.x * gridDim.y * gridDim.z = # of blocks being launched

- blockDim ⇒ blockDim.x * blockDim.y * blockDim.z = # of threads per block

- total threads = (threads per block) \* # of blocks

- The execution configuration (of a global function call) is specified by inserting an expression of the form `<<<gridDim, blockDim, Ns, S>>>`, where:

  - Dg (dim3) specifies the dimension and size of the grid.
  - Db (dim3) specifies the dimension and size of each block
  - Ns (size_t) specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory. (typically omitted)
  - S (cudaStream_t) specifies the associated stream, is an optional parameter which defaults to 0.

> source -> https://stackoverflow.com/questions/26770123/understanding-this-cuda-kernels-launch-parameters

## Thread Synchronization

- `cudaDeviceSynchronize();` ⇒ makes sure all the kernel for one problem are caught up so you can safely begin the next. Think of this as a barrier. Called from your `int main() {}` or another non`__global__` function.

- `__syncthreads();` to put a barrier for thread execution **inside** the kernel. useful if you are messing with the same memory spots and needs all the other jobs to catch up before you start making edits to a certain place. for example: one worker might be halfway done doing stuff to a place in memory. another worker might already be done the job task that the first worker is still doing. if this faster worker messes with a piece of memory that the slower worker still needs, you can get numerical instability and errors.

- `__syncwarps();` sync all threads within a warp

- why do we even need to synchronize threads? because threads are asynchronous and can be executed in any order. if you have a thread that is dependent on another thread, you need to make sure that the thread that is dependent on the other thread is not executed before the other thread is done.

- For example, if we want to vector add the two arrays `a = [1, 2, 3, 4]`, `b = [5, 6, 7, 8]` and store the result in `c`, then add 1 to each element in `c`, we need to ensure all the multiply operations catch up before moving onto adding (following PEDMAS). If we don't sync threads here, there is a possibility that we may get an incorrect output vector where a 1 is added before a multiply.

- A more clear but less common example would be when we parallelize a bit shift. If we have a bit shift operation that is dependent on the previous bit shift operation, we need to make sure that the previous bit shift operation is done before we move onto the next one.
  ![](../assets/bitshift1.png)

![](../assets/barrier.png)

## Thread Safety

- [Is CUDA thread-safe?](https://forums.developer.nvidia.com/t/is-cuda-thread-safe/2262/2)
- when a piece of code is “thread-safe” it can be run by multiple threads at the same time
  without leading to race conditions or other unexpected behaviour.

- race conditions are where one thread starts the next task before another finishes.
  to prevent race conditions, we use a special function called `cudaDeviceSynchronize()`
  to ensure all threads are caught up before giving them a new instruction to work on.
  think about a bunch of threads racing to the finish line, some finish before others
  for some reason and you have to manually tell those “winner” threads to wait at the
  finish line for the laggards.

- if you are wondering about calling multiple GPU kernels with different CPU threads,
  refer to the link above.

## SIMD/SIMT (Single Instruction, Multiple Threads)

- [Can CUDA use SIMD instructions?](https://stackoverflow.com/questions/5238743/can-cuda-use-simd-extensions)
- similar to CPU SIMD (single instruction multiple data), we have single instruction multiple thread on GPU.
- Instead of running the for loop sequentially, each thread can run a single iteration of the for loop so that it appears to only take the time of one iteration. it can grow linearly if you add more and more iterations as you would expect (not enough cores to parallel process all the independent iterations of the for loop)
- Simpler than CPU
  - in-order instruction issue
  - no branch prediction
  - significantly less control than CPU architecture gives us more room for more CORES

> Later on in the course (matmul optimization chapter), we will come back to optimzations connected to these special warp ops.
> ![Warp Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy tells us "There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited memory resources of that core. On current GPUs, a thread block may contain up to 1024 threads.", indicating 1024 threads per block, 32 threads per warp, and 32 warps per block is our theoretical limit.

## Math intrinsics
- device-only hardware instructions for fundamental math operations
- https://docs.nvidia.com/cuda/cuda-math-api/index.html
- you can use host designed operations like `log()` (host) instead of `logf()` (device) but they will run slower. these math essentials allow very math math operations on the device/GPU. you can pass in `-use_fast_math` to the nvcc compiler to convert to these device only ops at the cost of barely noticeable precision error.
- `--fmad=true` for fused multiply-add