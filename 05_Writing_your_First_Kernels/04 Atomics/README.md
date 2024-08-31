# What are Atomic Operations
    
by “atomic” we are referring to the indivisibility concept in physics where a thing cannot be broken down further.

An **atomic operation** ensures that a particular operation on a memory location is completed entirely by one thread before another thread can access or modify the same memory location. This prevents race conditions.

Since we limit the amount of work done on a single piece of memory per unit time throughout an atomic operation, we lose slightly to speed. It is hardware guaranteed to be memory safe at a cost of speed.

### **Integer Atomic Operations**

- **`atomicAdd(int* address, int val)`**: Atomically adds `val` to the value at `address` and returns the old value.
- **`atomicSub(int* address, int val)`**: Atomically subtracts `val` from the value at `address` and returns the old value.
- **`atomicExch(int* address, int val)`**: Atomically exchanges the value at `address` with `val` and returns the old value.
- **`atomicMax(int* address, int val)`**: Atomically sets the value at `address` to the maximum of the current value and `val`.
- **`atomicMin(int* address, int val)`**: Atomically sets the value at `address` to the minimum of the current value and `val`.
- **`atomicAnd(int* address, int val)`**: Atomically performs a bitwise AND of the value at `address` and `val`.
- **`atomicOr(int* address, int val)`**: Atomically performs a bitwise OR of the value at `address` and `val`.
- **`atomicXor(int* address, int val)`**: Atomically performs a bitwise XOR of the value at `address` and `val`.
- **`atomicCAS(int* address, int compare, int val)`**: Atomically compares the value at `address` with `compare`, and if they are equal, replaces it with `val`. The original value is returned.

### **Floating-Point Atomic Operations**

- **`atomicAdd(float* address, float val)`**: Atomically adds `val` to the value at `address` and returns the old value. Available from CUDA 2.0.
- Note: Floating-point atomic operations on double precision variables are supported starting from CUDA Compute Capability 6.0 using `atomicAdd(double* address, double val)`.

### From Scratch

Modern GPUs have special hardware instructions to perform these operations efficiently. They use techniques like Compare-and-Swap (CAS) at the hardware level.

You can think of atomics as a very fast, hardware-level mutex operation. It's as if each atomic operation does this:

1. lock(memory_location)
2. old_value = *memory_location
3. *memory_location = old_value + increment
4. unlock(memory_location)
5. return old_value

```cpp
__device__ int softwareAtomicAdd(int* address, int increment) {
    __shared__ int lock;
    int old;
    
    if (threadIdx.x == 0) lock = 0;
    __syncthreads();
    
    while (atomicCAS(&lock, 0, 1) != 0);  // Acquire lock
    
    old = *address;
    *address = old + increment;
    
    __threadfence();  // Ensure the write is visible to other threads
    
    atomicExch(&lock, 0);  // Release lock
    
    return old;
}
```


- Mutual Exclusion ⇒ https://www.youtube.com/watch?v=MqnpIwN7dz0&t
- "Mutual":
    - Implies a reciprocal or shared relationship between entities (in this case, threads or processes).
    - Suggests that the exclusion applies equally to all parties involved.
- "Exclusion":
    - Refers to the act of keeping something out or preventing access.
    - In this context, it means preventing simultaneous access to a resource.


```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// Our mutex structure
struct Mutex {
    int *lock;
};

// Initialize the mutex
__host__ void initMutex(Mutex *m) {
    cudaMalloc((void**)&m->lock, sizeof(int));
    int initial = 0;
    cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

// Acquire the mutex
__device__ void lock(Mutex *m) {
    while (atomicCAS(m->lock, 0, 1) != 0) {
        // Spin-wait
    }
}

// Release the mutex
__device__ void unlock(Mutex *m) {
    atomicExch(m->lock, 0);
}

// Kernel function to demonstrate mutex usage
__global__ void mutexKernel(int *counter, Mutex *m) {
    lock(m);
    // Critical section
    int old = *counter;
    *counter = old + 1;
    unlock(m);
}

int main() {
    Mutex m;
    initMutex(&m);
    
    int *d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple threads
    mutexKernel<<<1, 1000>>>(d_counter, &m);
    
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Counter value: %d\n", result);
    
    cudaFree(m.lock);
    cudaFree(d_counter);
    
    return 0;
}
```