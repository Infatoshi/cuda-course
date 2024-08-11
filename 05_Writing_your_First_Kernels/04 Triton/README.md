# Triton

## Design

- CUDA -> scalar program + blocked threads
- Triton -> blocked program + scalar threads

![](../assets/triton1.png)
![](../assets/triton2.png)

Why does this actually mean on an intuitive level?

- higher level of abstract for deep learning operations (activations functions, convolutions, matmul, etc)
- the compiler will take care of boilerplate complexities of load and store instructions, tiling, SRAM caching, etc
- python programmers can write kernels comparable to cuBLAS, cuDNN (very difficult for most CUDA/GPU programmers)

So can't we just skip CUDA and go straight to Triton.

- you may want to optimize your own kernels in CUDA
- you need to understand the paradigms used in CUDA and related topics to understand how to build on top of triton.

> Resources: [Paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), [Docs](https://triton-lang.org/main/index.html), [OpenAI Blog Post](https://openai.com/index/triton/), [Github](https://github.com/triton-lang/triton)
