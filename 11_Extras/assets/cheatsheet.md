## Cheatsheet 

## Host device memory transfer

1. **cudaMalloc((void**)&d_array, size * sizeof(type));**
   - Allocates memory on the GPU device.
   - `d_array` is a pointer to the device memory allocated with the specified size.

2. **cudaMemcpy(d_array, h_array, size * sizeof(type), cudaMemcpyHostToDevice);**
   - Copies data from the host array `h_array` to the device array `d_array`.
   - The copy is from host to device, with the specified size.

3. **cudaMemcpy(h_array, d_array, size * sizeof(type), cudaMemcpyDeviceToHost);**
   - Copies data from the device array `d_array` to the host array `h_array`.
   - The copy is from device to host, with the specified size.

4. **cudaFree(d_array);**
   - Frees the memory allocated on the device for `d_array`.
   - Important to prevent memory leaks and release resources.

5. **cudaMallocHost((void**)&h_array, size * sizeof(type));**
   - Allocates pinned (page-locked) memory on the host.
   - Pinned memory allows faster transfer to and from the device.

6. **cudaFreeHost(h_array);**
   - Frees the pinned host memory allocated with `cudaMallocHost`.
   - Should be called after using the pinned memory to release resources.

7. **cudaMemcpyAsync(dst, src, size, kind, stream);**
   - Performs an asynchronous memory copy between src and dst.
   - The copy is associated with a specific stream, allowing overlap with kernel execution.

## Kernel launch syntax

8. `__global__ void kernelName(parameters) { /* kernel code */ }`
   - Defines a CUDA kernel function to be executed on the GPU.
   - The `__global__` specifier indicates that this function is launched on the device.

9. `kernelName<<<gridDim, blockDim>>>(arguments);`
   - Launches the kernel on the GPU with specified grid and block dimensions.
   - `gridDim` and `blockDim` define the number of blocks and threads per block, respectively.

10. `kernelName<<<gridDim, blockDim, sharedMemSize>>>(arguments);`
    - Launches the kernel with an additional specification for shared memory per block.
    - `sharedMemSize` sets the amount of shared memory allocated for each block in bytes.

11. `kernelName<<<gridDim, blockDim, sharedMemSize, stream>>>(arguments);`
    - Launches the kernel with shared memory and assigns it to a specific CUDA stream.
    - Using streams allows for asynchronous execution of kernels and memory operations.

## Thread indexing

### 1D Indexing

12. **int tid = threadIdx.x;**
   - Retrieves the x-coordinate of the thread's index within its block.
   - Useful for identifying the thread's position within the block.

13. **int bid = blockIdx.x;**
   - Retrieves the x-coordinate of the block's index within the grid.
   - Helps in determining the block's position within the grid.

14. **int idx = blockIdx.x * blockDim.x + threadIdx.x;**
   - Calculates the global thread index in a 1D grid.
   - Combines block and thread indices to address data in a linear fashion.

### 2D Indexing

15. **int row = blockIdx.y * blockDim.y + threadIdx.y;**
   - Computes the row index by combining block and thread indices in the y-dimension.
   - Essential for 2D data processing, such as image processing.

16. **int col = blockIdx.x * blockDim.x + threadIdx.x;**
   - Computes the column index by combining block and thread indices in the x-dimension.
   - Helps in addressing elements in a 2D grid.

17. **int idx_2d = row * width + col;**
   - Converts 2D row and column indices to a linear index.
   - Assumes row-major order, commonly used in arrays.

### 3D Indexing

18. **int idx_3d = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;**
   - Computes a linear index from 3D block and thread indices.
   - Necessary for handling 3D data, such as volumetric data.

19. **Considerations for 3D indexing:**
   - Ensure grid and block dimensions are correctly set.
   - Be aware of memory layout to optimize data access.


## Functions

20. **__global__ void functionName() { }**
	- Declares a CUDA kernel function.
	- Executed on the GPU, called from the host.

21. **__device__ void functionName() { }**

	- Declares a function to run on the GPU.
	- Can only be called by other device functions or kernels.

22. **__host__ void functionName() { }**

	- Declares a function to run on the CPU.
	- Can only be called from host code.

23. **__noinline__ void functionName() { }**

	- Instructs the compiler not to inline the function.
	- Prevents unwanted inlining optimizations.

24. **__forceinline__ void functionName() { }**

	- Instructs the compiler to inline the function.
	- Reduces function call overhead by inlining code.

## Variable type Qualifiers


24. **__shared__ type variableName;**
   - Declares a variable in shared memory.
   - Shared memory is fast on-chip memory accessible by all threads in a block.

25. **__device__ type variableName;**
   - Declares a variable in device global memory.
   - Accessible by all threads across the entire grid.

26. **__constant__ type variableName;**
   - Declares a variable in constant memory.
   - Constant memory is cached and optimized for read-only access by all threads.

27. **__managed__ type variableName;**
   - Declares a variable in managed memory (Unified Memory).
   - Accessible by both the host (CPU) and device (GPU) without explicit memory transfers.

## Thread Synchronization

28. **__syncthreads();**
   - Synchronizes all threads within a block.
   - Ensures all threads in the block reach this point before proceeding.

29. **__syncthreads_and(predicate);**
   - Synchronizes threads within a block and evaluates a predicate.
   - Returns true if all threads in the block satisfy the predicate.

30. **__syncthreads_or(predicate);**
   - Synchronizes threads within a block and evaluates a predicate.
   - Returns true if at least one thread in the block satisfies the predicate.

31. **__syncthreads_count(predicate);**
   - Synchronizes threads within a block and counts the number of threads satisfying the predicate.
   - Returns the count of threads where the predicate is true.

32. **__threadfence();**
   - Ensures all memory writes by the calling thread are visible to all threads in the device.
   - Acts as a device-wide memory fence.

33. **__threadfence_block();**
   - Ensures all memory writes by the calling thread are visible to all threads in the block.
   - Acts as a block-wide memory fence.

34. **__threadfence_system();**
   - Ensures all memory writes by the calling thread are visible to all threads in the system (including host and other devices).
   - Acts as a system-wide memory fence.

## Stream Synchoronization


35. **cudaStreamSynchronize(stream);**
   - Blocks the host until all operations in the specified stream are complete.
   - Ensures synchronization with a specific CUDA stream.

36. **cudaDeviceSynchronize();**
   - Blocks the host until all operations on the device (GPU) are complete.
   - Ensures full device synchronization.

37. **cudaStreamWaitEvent(stream, event);**
   - Makes the specified stream wait until the given event is recorded.
   - Synchronizes streams based on events for fine-grained control.


## Atomic Operations


38. **atomicAdd(address, val);**
   - Atomically adds `val` to the value at `address`.
   - Returns the original value at `address`.

39. **atomicSub(address, val);**
   - Atomically subtracts `val` from the value at `address`.
   - Returns the original value at `address`.

40. **atomicExch(address, val);**
   - Atomically exchanges the value at `address` with `val`.
   - Returns the original value at `address`.

41. **atomicMin(address, val);**
   - Atomically sets the value at `address` to the minimum of its current value and `val`.
   - Returns the original value at `address`.

42. **atomicMax(address, val);**
   - Atomically sets the value at `address` to the maximum of its current value and `val`.
   - Returns the original value at `address`.

43. **atomicInc(address, val);**
   - Atomically increments the value at `address` by 1, wrapping around to 0 if it exceeds `val`.
   - Returns the original value at `address`.

44. **atomicDec(address, val);**
   - Atomically decrements the value at `address` by 1, wrapping around to `val` if it goes below 0.
   - Returns the original value at `address`.

45. **atomicCAS(address, compare, val);**
   - Atomically compares the value at `address` with `compare`. If they match, sets the value at `address` to `val`.
   - Returns the original value at `address`.

## Error Handlin

46. **cudaError_t error = cudaGetLastError();**
   - Retrieves the last error from a CUDA runtime call.
   - Useful for checking errors after asynchronous operations.

47. **const char* errorString = cudaGetErrorString(error);**
   - Converts a CUDA error code (`cudaError_t`) into a human-readable string.
   - Helps in debugging by providing descriptive error messages.

48. **Error Checking Macro:**
   ```cpp
   #define CUDA_CHECK(call) { \
       cudaError_t error = call; \
       if (error != cudaSuccess) { \
           printf("CUDA error: %s\n", cudaGetErrorString(error)); \
           exit(1); \
       } \
   }
   ```
   - A macro to wrap CUDA calls and automatically check for errors.
   - Prints the error message and exits the program if an error occurs.

## Device Management 


49. **cudaDeviceProp prop; cudaGetDeviceProperties(&prop, deviceId);**
   - Retrieves properties of the specified CUDA device (e.g., compute capability, memory size).
   - Stores the properties in the `cudaDeviceProp` structure.

50. **cudaSetDevice(deviceId);**
   - Sets the specified device (`deviceId`) as the current device for CUDA operations.
   - Useful for multi-GPU systems to select a specific GPU.

51. **int deviceId; cudaGetDevice(&deviceId);**
   - Retrieves the ID of the currently active CUDA device.
   - Stores the device ID in the provided variable.

52. **int deviceCount; cudaGetDeviceCount(&deviceCount);**
   - Retrieves the total number of CUDA-capable devices available on the system.
   - Stores the count in the provided variable.

## Stream Management 


53. **cudaStream_t stream; cudaStreamCreate(&stream);**
   - Creates a CUDA stream for asynchronous operations.
   - Initializes the stream with default behavior (blocking with respect to the host).

54. **cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);**
   - Creates a CUDA stream with specific flags (e.g., `cudaStreamNonBlocking`).
   - Non-blocking streams do not synchronize with the default stream.

55. **cudaStreamDestroy(stream);**
   - Destroys a CUDA stream and releases associated resources.
   - Ensures proper cleanup of streams after use.

56. **cudaStreamSynchronize(stream);**
   - Blocks the host until all operations in the specified stream are complete.
   - Ensures synchronization with the stream before proceeding.

## Event Manager

57. **cudaEvent_t event; cudaEventCreate(&event);**
   - Creates a CUDA event for synchronization and timing.
   - Initializes the event for use in measuring GPU operations.

58. **cudaEventRecord(event, stream);**
   - Records an event in the specified stream.
   - Marks a point in the stream for synchronization or timing.

59. **cudaEventSynchronize(event);**
   - Blocks the host until the specified event is recorded.
   - Ensures synchronization with the event.

60. **cudaEventElapsedTime(&ms, start, stop);**
   - Calculates the elapsed time (in milliseconds) between two events (`start` and `stop`).
   - Useful for measuring the duration of GPU operations.

61. **cudaEventDestroy(event);**
   - Destroys a CUDA event and releases associated resources.
   - Ensures proper cleanup of events after use.

## Some more useful macros and constants

```c
// Block size macros
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Grid calculation macro
#define GRID_SIZE(n, b) ((n + b - 1) / b)

// Maximum grid and block dimensions
dim3 maxGridSize(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
dim3 maxThreadsPerBlock(prop.maxThreadsPerBlock);
```

62. **#define BLOCK_SIZE 256**
   - Defines the number of threads per block (block size).
   - Commonly set to a multiple of the warp size (e.g., 256).

63. **#define WARP_SIZE 32**
   - Defines the warp size, which is 32 threads for all CUDA-capable devices.
   - Useful for warp-level optimizations.

64. **#define GRID_SIZE(n, b) ((n + b - 1) / b)**
   - Calculates the grid size required to process `n` elements with `b` threads per block.
   - Ensures all elements are covered, even if `n` is not a multiple of `b`.

65. **dim3 maxGridSize(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);**
   - Retrieves the maximum grid dimensions from the device properties (`cudaDeviceProp`).
   - Useful for ensuring grid dimensions do not exceed hardware limits.

66. **dim3 maxThreadsPerBlock(prop.maxThreadsPerBlock);**
   - Retrieves the maximum number of threads per block from the device properties.
   - Ensures block dimensions do not exceed hardware limits.


## Common Runtime API Functions



67. **cudaDeviceReset();**
   - Resets the current CUDA device, cleaning up all resources (e.g., memory, streams, events).
   - Useful for ensuring a clean state during debugging or reinitialization.

68. **cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);**
   - Configures the cache preference for a kernel to prioritize shared memory.
   - Optimizes performance for workloads that benefit from larger shared memory.

69. **cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);**
   - Configures the cache preference for a kernel to prioritize L1 cache.
   - Optimizes performance for workloads that benefit from larger L1 cache.

70. **cudaDeviceSetLimit(cudaLimitStackSize, value);**
   - Sets the stack size limit per thread for the current device.
   - Useful for increasing stack size if kernel recursion or large local variables are used.

71. **cudaDeviceSetLimit(cudaLimitMallocHeapSize, value);**
   - Sets the heap size limit for dynamic memory allocation (`malloc`/`new`) on the device.
   - Useful for increasing heap size if large dynamic allocations are required.

## Template functions

72. **Template Kernel Definition:**
   ```cpp
   template<typename T>
   __global__ void kernelName(T* data) {
       // kernel code
   }
   ```
   - Defines a CUDA kernel with a template parameter `T`.
   - Allows the kernel to work with different data types (e.g., `float`, `int`).

73. **Launching a Template Kernel:**
   ```cpp
   kernelName<float><<<grid, block>>>(data);
   ```
   - Launches the template kernel with a specific type (e.g., `float`).
   - Specifies grid and block dimensions for kernel execution.

## Warp Level Operations

### Warp Vote Functions
74. **__all_sync(mask, predicate);**
   - Evaluates `predicate` for all active threads in the warp (specified by `mask`).
   - Returns true if **all** threads in the warp satisfy the predicate.

75. **__any_sync(mask, predicate);**
   - Evaluates `predicate` for all active threads in the warp (specified by `mask`).
   - Returns true if **any** thread in the warp satisfies the predicate.

76. **__ballot_sync(mask, predicate);**
   - Evaluates `predicate` for all active threads in the warp (specified by `mask`).
   - Returns a 32-bit mask where each bit represents whether a thread satisfies the predicate.

### Warp Shuffle Functions
77. **__shfl_sync(mask, var, srcLane);**
   - Shuffles the value `var` from the thread with lane ID `srcLane` to all active threads in the warp.
   - Requires all threads in the warp to participate (synchronized using `mask`).

78. **__shfl_up_sync(mask, var, delta);**
   - Shuffles the value `var` from the thread with lane ID `(current_lane - delta)` to the current thread.
   - Threads with lane IDs less than `delta` receive their own value.

79. **__shfl_down_sync(mask, var, delta);**
   - Shuffles the value `var` from the thread with lane ID `(current_lane + delta)` to the current thread.
   - Threads with lane IDs greater than or equal to `(32 - delta)` receive their own value.

80. **__shfl_xor_sync(mask, var, laneMask);**
   - Shuffles the value `var` between threads whose lane IDs are XORed with `laneMask`.
   - Useful for butterfly-style data exchange within the warp.

### Notes:
- **`mask`**: A 32-bit mask specifying which threads in the warp are active.
- **`predicate`**: A condition evaluated by each thread.
- **`var`**: The variable to be shuffled or evaluated.
- These functions require CUDA 9.0 or later and are used for warp-level parallelism and communication.
