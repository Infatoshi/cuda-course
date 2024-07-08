# CUDA Course

GitHub Repo for CUDA Course on FreeCodeCamp

## Table of Contents

1. [Course Philosophy](#course-philosophy)
2. [Overview](#overview)
3. [Prerequisites](#prerequisites)
4. [Course Structure](#course-structure)
5. [Key Takeaways](#key-takeaways)
6. [Hardware Requirements](#hardware-requirements)
7. [Use Cases for CUDA/GPU Programming](#use-cases-for-cudagpu-programming)
8. [Resources](#resources)

## Course Philosophy

This course aims to:

- Lower the barrier to entry for HPC jobs
- Provide a foundation for understanding projects like Karpathy's [llm.c](https://github.com/karpathy/llm.c)
- Consolidate scattered CUDA programming resources into a comprehensive, organized course

## Overview

- Focus on GPU kernel optimization for performance improvement
- Cover CUDA, PyTorch, and Triton
- Emphasis on technical details of writing faster kernels
- Brief introduction to deployment in the deep learning ecosystem
- Tailored for NVIDIA GPUs
- Culminates in a simple CNN project in CUDA

## Prerequisites

- Python programming (required)
- Basic differentiation and vector calculus for backprop (recommended)
- Linear algebra fundamentals (recommended)

## Key Takeaways

- Optimizing existing implementations
- Building CUDA kernels for cutting-edge research
- Understanding GPU performance bottlenecks, especially memory bandwidth

## Hardware Requirements

- Any NVIDIA GTX, RTX, or datacenter level GPU
- Cloud GPU options available for those without local hardware

## Use Cases for CUDA/GPU Programming

- Deep Learning (primary focus of this course)
- Graphics and Ray-tracing
- Fluid Simulation
- Video Editing
- Crypto Mining
- 3D modeling
- Anything that requires parallel processing with large arrays

## Resources

- GitHub repo (this repository)
- Stack Overflow
- NVIDIA Developer Forums
- NVIDIA and PyTorch documentation
- LLMs for navigating the space

> Note: This course is designed for Ubuntu Linux. Windows users can use Windows Subsystem for Linux or Docker containers.

## TODO

- Revamp DL ecosystem notes
- Struct pointers: `nn.w1` vs `nn->w1`
- NCCL & cuBLAS-Xt fundamental operations
- PyTorch extensions in C++
- NVTX profiler
- MLIR vs LLVM