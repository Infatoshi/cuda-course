# CUDA Basics

## Lets print out some stats about your GPU
![](../assets/gpustats.png)


## Easy stuff
Host ⇒ CPU ⇒ Uses RAM sticks on the motherboard

Device ⇒ GPU ⇒ Uses on Chip VRAM (video memory for desktop PCs)

CUDA program surface level runtime:

1. copy input from host to device
2. load GPU program and execute using the transferred on-device data
3. copy results from device back to host so you can display/use it somehow

## Device VS Host naming scheme
`h_A` refers to host (CPU) for variable name “A”

`d_A` refers to device (GPU) for variable name “A” 

`__global__` is visible globally, meaning the CPU or  *host* can call these global functions. these don’t typically return anything but just do really fast operations to a variable you pass in. for example, I could multiply matrix A and B together, but I need to pass in a matrix of the needed size as C and change the values in C to the outputs of A * B matmul. these are your cuda kernels 

`__device__` is a very cool function I haven’t dived into yet but this is the small job that only the GPU can call. GPT-4 really liked my example of having a raw attention score matrix living on the `__global__` gpu cuda kernel and it needs to apply a scalar mask. instead of also doing this in the cuda kernel, we can have a `__device__` function defined in another .cu file or just exist as a function in the same file that does this SIMD scalar masking on any matrix we give it. this is the cuda equivalent of calling a function in a library instead of writing the function in your `main.py` file

`__host__` is only going to run on CPU. same as running a regular c/c++ script on CPU without cuda.

## Memory Management

- `cudaMalloc` memory allocation on VRAM only (also called global memory)

```
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, N*N*sizeof(float));
    cudaMalloc(&d_b, N*N*sizeof(float));
    cudaMalloc(&d_c, N*N*sizeof(float));
```

- `cudaMemcpy` can copy from device to host, host to device, or device to device (edge cases)
    - host to device ⇒ CPU to GPU
    - device to host ⇒ GPU to CPU
    - device to device ⇒ GPU location to different GPU location
    - **`cudaMemcpyHostToDevice`**, **`cudaMemcpyDeviceToHost`**, or **`cudaMemcpyDeviceToDevice`**
- `cudaFree` will free memory on the device

# `nvcc` compiler
- Host code
    - modifed to run kernels
    - compiled to x86 binary

- Device code
    - compiled to PTX (parallel thread execution)

    - stable across multiple GPU generations

- JIT (just-in-time)

    - PTX into native GPU instructions

    - allows for forward compatibility

## CUDA Hierarchy?
1. Kernel executes in a thread
2. Threads grouped into Thread Blocks (aka Blocks)
3. Blocks grouped into a Grid
4. Kernel executed as a Grid of Blocks of Threads

### 4 technical terms:
- `gridDim` ⇒ number of blocks in the grid
- `blockIdx` ⇒ index of the block in the grid
- `blockDim` ⇒ number of threads in a block
- `threadIdx` ⇒ index of the thread in the block

(more on this in video lectures)

## Threads
- each thread has local memory (registers) and is private to the thread
- if want to add `a = [1, 2, 3, ... N]` and `b = [2, 4, 6, ... N]` each thread would do a single add ⇒ `a[0] + b[0]` (thread 1); `a[1] + b[1]` (thread 2); etc...

## Warps
![](../assets/weft.png)
- https://en.wikipedia.org/wiki/Warp_and_weft
- The warp is the set of [yarns](https://en.wikipedia.org/wiki/Yarn) or other things stretched in place on a [loom](https://en.wikipedia.org/wiki/Loom) before the weft is introduced during the weaving process. It is regarded as the *longitudinal* set in a finished fabric with two or more sets of elements.
- Each warp is inside of a block and parallelizes 32 threads
- Instructions are issued to warps that then tell the threads what to do (not directly sent to threads)
- There is no way of getting around using warps
- Warp scheduler makes the warps run
- 4 warp schedulers per SM
![](../assets/schedulers.png)

## Blocks
- each block has shared memory (visible to all threads in thread block)
- execute the same code on different data, shared memory space, more efficient memory reads and writes since coordination is better

## Grids
- during kernel execution, the threads within the blocks within the grid can access global memory (VRAM)
- contain a bunch of blocks. best example is grids handle batch processing, where each block in the grid is a batch element

> why not just use only threads instead of blocks and threads? add to this given our knowledge of how warps group and execute a batch of 32 threads in lockstep
> Logically, this shared memory is partitioned among the blocks. This means that a thread can communicate with the other threads in its block via the shared memory chunk. 

- CUDA parallelism is scalable because their aren’t sequential block run-time dependencies.What I mean here is that you may not run Block 0 & Block 1, then Block 2 & 3… It may be Block 3 & 0, then Block 6 & 1. This means each of these mini “jobs” are solving a subset of the problem independent of the others. Like one piece of the puzzle. As long as all the pieces are assembled in the right place at the end, it works!

> [How do threads map onto CUDA cores?](https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores)