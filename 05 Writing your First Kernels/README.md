# Writing your first CUDA Kernels

> Everything starts here -> https://docs.nvidia.com/cuda/
> We mainly focus on the CUDA C programming guide -> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
> Consider following along here -> https://developer.nvidia.com/blog/even-easier-introduction-cuda/

- its generally a good idea to write code for a kernel first on CPU (easy to write), then on GPU to ensure your logic lines up on the level of blocks and threads. you can set some input x, feed it through the CPU function and GPU kernel, check if outputs are the same. this tells you if your GPU code is working as expected

- Practice vector addition and matrix multiplication by hand
- Understand the concept of threads, blocks, and grids

## To run our compile & run our vec add kernel:

```bash
nvcc -o 01 01_vector_addition.cu
./01
```

(add small explanations and diagrams from assets folder)

## Hardware Mapping

- CUDA cores handle threads
- Streaming Multiprocessors (SMs) handle blocks (typically multiple blocks per SM depending on resources required)
- Grids are mapped to the entire GPU since they are the highest level of the hierarchy

## Memory Model

- Registers & Local Memory
- Shared Memory ⇒ allows threads within a block to communicate
- L2 cache. acts as buffer between cores/registers and global mem. also is a shared memory across SMs
- L2 cache and Shared/L1 cache both use the same circuitry as SRAM so they run at about the same speed. L2 cache is bigger and gives
- Speed: While both use SRAM, L2 is generally slower than L1. This is not due to the underlying technology, but rather due to:
  - Size: L2 is larger, which increases access time.
  - Shared nature: L2 is shared among all SMs, requiring more complex access mechanisms.
  - Physical location: L2 is typically further from the compute units than L1.
- Global Memory ⇒ Stores data copies to and from Host. Everything on device can access Global mem
- Host ⇒ 16/32/64GB DRAM depending on your rig (those 4 RAM sticks on the motherboard)
- Arrays too big to fit into the Register will spill into local memory. our goal is to make sure this doesn’t happen because we want to keep our program running as fast as possible

![](assets/memhierarchy.png)


### What is _random_ access memory?

- in a video tape you have to access the bits sequentially to reach
  the last ones. random refers to the nature of instantly getting information
  from a given random index (without relying on having to index anything else). we are provided with an abstraction that seems like memory is a giant line but on chip its actually layed out as a grid (circuitry takes care of things here)

![](../assets/memmodel.png)


> [Efficient Matrix Tranpose Nvidia Blog Post](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

