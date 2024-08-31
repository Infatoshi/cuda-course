#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

// Kernel without atomics (incorrect)
__global__ void incrementCounterNonAtomic(int* counter) {
    // not locked
    int old = *counter;
    int new_value = old + 1;
    // not unlocked
    *counter = new_value;
}

// Kernel with atomics (correct)
__global__ void incrementCounterAtomic(int* counter) {
    int a = atomicAdd(counter, 1);
}

int main() {
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtomic;

    // Allocate device memory
    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));

    // Copy initial counter values to device
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernels
    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    // Copy results back to host
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Non-atomic counter value: %d\n", h_counterNonAtomic);
    printf("Atomic counter value: %d\n", h_counterAtomic);

    // Free device memory
    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    return 0;
}