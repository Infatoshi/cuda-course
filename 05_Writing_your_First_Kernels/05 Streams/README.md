# CUDA Streams Examples

## Intuition
You can think of streams as "river streams" where the direction of operations flows only forward in time (like a timeline). For example, copy some data over (time step 1), then do some computation (time step 2), then copy some data back (time step 3). This is the basic idea behind streams. 

We can have multiple streams at once in CUDA, and each stream can have its own timeline. This allows us to overlap operations and make better use of the GPU.

When training a massive language model, it would be silly to spend a ton of time loading all the tokens in and out of the GPU. Streams allow us to move data around while also doing computation at all times. Streams introduce a software abstraction called "prefetching", which is a way to move data around before it is needed. This is a way to hide the latency of moving data around. 

This project demonstrates the usage of CUDA streams for concurrent execution and better GPU utilization. It contains two examples:


## Code Snippets
- default **stream** = **stream** 0 = null **stream**
```cpp
// This kernel launch uses the null stream (0)
myKernel<<<gridSize, blockSize>>>(args);

// This is equivalent to
myKernel<<<gridSize, blockSize, 0, 0>>>(args);
```

Remember this part from the Kernels section?
- The execution configuration (of a global function call) is specified by inserting an expression of the form `<<<gridDim, blockDim, Ns, S>>>`, where:

  - Dg (dim3) specifies the dimension and size of the grid.
  - Db (dim3) specifies the dimension and size of each block
  - Ns (size_t) specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory. (typically omitted)
  - S (cudaStream_t) specifies the associated stream, is an optional parameter which defaults to 0.

- stream 1 and stream 2 are created with different priorities. this means they are executed in a certain order at runtime. this essentially gives us more control over the concurrent execution of our kernels.

```cpp
    // Create streams with different priorities
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
```

## Examples

1. `stream_basics.cu`: Illustrates basic stream usage with asynchronous memory transfers and kernel launches.
2. `stream_advanced.cu`: Demonstrates more advanced concepts like stream priorities, callbacks, and inter-stream dependencies.

## Compilation

To compile the examples, use the following commands:

```bash
nvcc -o 01 01_stream_basics.cu
nvcc -o 02 02_stream_advanced.cu
```

## Docs
- https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

## Pinned Memory
- "we're gonna need this for later, so don't play with it" is a good way to think about it.
- pinned memory is memory that is locked in place and cannot be moved around by the OS. This is useful for when you want to move data to the GPU and do some computation on it. If the OS moves the data around, the GPU will be looking for the data in the wrong place and you will get a segfault.
```cpp
// Allocate pinned memory
float* h_data;
cudaMallocHost((void**)&h_data, size);
```

## Events

## Callbacks
-  By using callbacks, you can set up a pipeline where the completion of one operation on the GPU triggers the start of another operation on the CPU, which might then queue more work for the GPU. (as seen in the nvidia concurrency docs above)

