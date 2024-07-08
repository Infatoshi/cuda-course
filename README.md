# cuda-course

Github Repo for CUDA Course on FreeCodeCamp

## Table of Contents

1. Course Philosophy(link)
2. Intro

## Course Philosophy

- Why did I create this course? ⇒ barrier to entry is very high for kernel engineer jobs OPENAI (https://openai.com/careers/gpu-kernels-engineer/) and NVIDIA High Performance Computing Engineering roles (https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/details/Senior-HPC-Performance-Engineer_JR1977468).
- generally speaking the point of writing GPU kernels or playing with code AT ALL on the gpu is to run something faster. take a nested for loop in python to do some linear algebra. your job is to use your knowledge GPU architecture, kernel launch configuations, and a bunch of other cool stuff we cover in this course to make that code run as fast as possible.
- I want to note early on that this course won’t be CUDA only. we reference pytorch a lot and even a bit of triton too. I will cover the the terms and ecosystem so it makes more sense as you progress through the material.
- We will not be writing extensive training tests to deploy a model, but rather focusing on the technical details of writing faster kernels. The deployment part will be one of the first parts covered in the course. It would be weird to show you how to write really fast kernels and not plant a seed on how you could use those the tools in the deep learning ecosystem to significantly cut costs.
- This is strictly for nvidia gpus, if you don’t have one, consider renting the cheapest ones in the cloud and playing around. I advise you look into the pricing before giving a definite “NO”. at first, I was surprised at how low the cost for some cloud services where, especially the non-compute demanding ones. you are just experimenting, not writing the training run for GPT-5
- We will soon go over the table of contents, in which we end off with a simple CNN project in CUDA. Instead of making a mini project on the common decoder-only transformer architecture, I want to cover something else so karpathy’s llm.c (https://github.com/karpathy/llm.c) will be a fluid transition after coming out of this. I’m not certain when or if lectures on llm.c will be released at all but to make this course worth your time, I cover many things, giving you the fundamentals to go and read llm.c on your own.
- As for course prerequisites, python programming will help in understanding what we are implementing in lower-level languages. Basic differentiation and vector calculus will make learning easier, but it really only required for intuition behind backpropagation. Linear algebra will make your life easier by not having to learn the fundamental algorithms from scratch, but I will go over them anyways. If you really care, review the following: matrix transpose, matrix multiplication, chain rule from calculus, gradient vs derivative… MAYBE MORE
- Nowadays, they say we have too much data, but very little cleaned data. I have taken everything from all the other videos/courses on youtube and elsewhere and put the must haves into a single course. This includes the topics covered by paid courses too. I have the links for youtube videos I’ve parsed through right here so you don’t have to spend hours doing the same.
- I suggest following this notion doc to maintain a structured learning approach. I may also include intuitive excalidraw drawings and diagrams to break down the complex of material in this course ⇒ which I will also provide links to (join my discord server to find the links). When you can visualize the geometrical interaction of the topics presented in this course, you will likely find parallel programming extremely fun.
- Some key takeaways from this course would be to take an existing implementation and make it faster, or to look at current research not yet done, and build a CUDA kernel to make it run as fast as possible. I will cover both the profiling and optimization aspect, as well as implementing neural network functions like backpropagation, batchnorm, convolutions, activations, and more.
- SPOILER ALERT… through experimentation and research, you will learn that the main GPU performance bottleneck is memory bandwidth. in deep learning, we have these giant inscrutable matrices that cannot fit into the on chip memory at once. we have to take little chunks of them at a time from off-chip memory. You may think the VRAM is fast, but its actually the main bottleneck of many deep learning GPU applications. it takes a really long time to copy/transfer data over to the cores from this off-chip VRAM. by off-chip, I mean the memory cells aren’t close to the cores, wheras SRAM and register memory is directly on chip (very close to cores). to illusrate, the SRAM has a memory bandwidth of about 20 terabytes a second, whereas VRAM is about 100 gigabytes per second. we end up with a bunch of super fast CUDA cores waiting for data to arrive rather than doing constantly computation.
- you can continue run with any NVIDIA GTX, RTX, or datacenter level GPU.
- Use cases for CUDA / Parallel / GPU Programming
  - Other use cases (opt. might omit later)
    - Graphics and Ray-tracing
      - Simply, **OpenGL draws everything on your screen really fast, OpenCL and CUDA process the calculations necessary when your videos interact with your effects and other media**.
    - Fluid Simulation
    - Video Editting
    - Crypto Mining
    - 3D modelling in Software like Blender
  - This course
    - You guessed it! …Deep Learning the #1 use case for CUDA is primarily what I’ll be covering in this course
  - Also building up GPU clusters for multi-node parallelism but their are docs for this
- check github, stackoverflow, nvidia developer forums, nvidia docs, pytorch docs if your issue is related to CUDA OR triton in pytorch, chatGPT or other LLMs to help navigate the space more easily (information won’t be as hard to process since its neatly organized). this is a part of being a programmer :)
- all the code and notes for this course are kept in the github repo in the description. the ecosystem changes over time so if you’re looking at this a couple years from now it might be outdated. I’ll make sure to publish working code to github repo regardless of what happens to the ecosystem. this may make it slightly confusing at times. but you should be able to reproduce all the material with the contents in the github repo.
- Heads up, I don’t cover anything about windows (even though it is possible). because I do everything on ubuntu linux. You can use Linux Subsystem for windows if you cannot switch to ubuntu ⇒ https://learn.microsoft.com/en-us/windows/wsl/install & https://docs.nvidia.com/cuda/wsl-user-guide/index.html. you can also fire up a ubuntu docker container and get access to all linux features there (without UIs)

## TODO:

- revamp DL ecosystem notes
- struct ptrs ⇒ `nn.w1` vs `nn->w1`
- NCCL & cuBLAS-Xt fundamental operations
  - cublas-xt vs cublasMP
- pytorch extensions C++
- nvtx profiler
- mlir vs llvm
