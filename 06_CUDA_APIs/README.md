# CUDA API 
> Includes cuBLAS, cuDNN, cuBLASmp

- the term “API” can be confusing at first. all this mean is we have a library where we can’t see the internals. there is documentation on the function calls within the API, but its a precompiled binary that doesn’t expose source code. the code is highly optimized but you can’t see it. (keep this here as it universally applies to all the libs/APIs listed below)

## Opaque Struct Types (CUDA API):
- you cannot see or touch the internals of the type, just external like names, function args, etc. `.so` (shared object) file referenced as an opaque binary to just run the compiled functions at high throughput. If you search up cuFFT, cuDNN, or any other CUDA extension, you will notice it comes as an API, the inability to see through to the assembly/C/C++ source code refers to usage of the word “opaque”. the struct types are just a general type in C that allows NVIDIA to build the ecosystem properly. cublasLtHandle_t is an example of an opaque type containing the context for a cublas Lt operation

If you’re trying to just figure out how to get the fastest possible inference to work on your cluster, you will need to understand the details under the hood. To navigate the CUDA API, I’d recommend using the following tricks:
1. [perplexity.ai](http://perplexity.ai) (most up to date information and will fetch data in real time)
2. google search (arguably worse than perplexity but its alright to take the classic approach to figuring things out)
3. chatGPT for general knowledge that’s less likely to be past its training cutoff
4. keyword search in nvidia docs


## Error Checking (API Specific)

- cuBLAS for example

```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- cuDNN example

```cpp
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- The need for error checking goes as follows: you have a context for a CUDA API call that you configure, then you call the operation, then you check the status of the operation by passing the API call into the "call" field in the macro. If it returns successful your code will continue running as expected. If it fails, you will get a descriptive error message instead of just a segmentation fault or silently incorrect result.
- There are obviously more error checking macros for other CUDA APIs, but these are the most common ones (needed for this course).
- Consider reading this guide here -> [Proper CUDA Error Checking](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/)


## Matrix Multiplication
- cuDNN implicitly supports matmul through specific convolution and deep learning operations but isn't presented as one of the main features of cuDNN
- You'll be best off using the deep learning linear algebra operations in cuBLAS for matrix multiplication since it has wider coverage and is tuned for high throughput matmul
> Side notes (present to show that its not that hard to transfer knowledge of, say, cuDNN to cuFFT with the way you configure and call an operation)

## Resources:
- [CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples)