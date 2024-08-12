**cuBLASmp** vs **NCCL** vs **MIG** (multi instance GPU)

## cuBLAS-mp
- **NVIDIA cublasMp** is a high performance, multi-process, GPU accelerated library for distributed basic dense linear algebra. cublasMP for multi-gpu, single node level tensor ops. use this if a model cant fit on a single instance.

## NCCL
- NVIDIA Collective Communications Library ⇒ for dist cluster computing 
- NCCL used for distributing information, collecting it, and acting as a general cluster level communicator. cublasMP is doing the grunt work of doing matmuls across 8xH100s and NCCL is going to run this in batches. remember “collective communications” ⇒ all-reduce, broadcast, gather, and scatter across multiple GPUs or nodes
- in pytorch, you’d use Distributed Data Parallel ⇒ https://pytorch.org/tutorials/intermediate/ddp_tutorial.html, but if you’re writing GPT-5’s training run you’d want to pay CUDA experts to squeeze every bit of performance out at the datacenter level.
- [CUDA MODE: NCCL Lecture](https://www.youtube.com/watch?v=T22e3fgit-A&ab_channel=CUDAMODE)
- [Extended GPU Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-gpu-memory)
- Model Parallelism (weights) VS Data parallelism (batches)
- setup here ⇒ https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
- we can see all the operations here ⇒ https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html

## MIG (Multi-Instance GPU)
- MIG ⇒ taking a big GPU and literally slicing it into smaller, independent GPUs
- datacenter usecases where you get more value splitting one node into a bunch of others (customers might not be maxing out the compute utilization)

