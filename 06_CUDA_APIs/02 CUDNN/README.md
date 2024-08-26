# cuDNN

you technically don’t need cuFFT or a ton of manually written custom kernels to write a GPT training run + inference. fast convolve is built into cuDNN, and cuBLAS matmul is included in cuDNN at greater abstraction. still a good idea to review the idea of slow conv, fast conv, slow matmul, fast matmul.

NVIDIA cuDNN provides highly tuned implementations of operations arising frequently in deep learning applications:

- Convolution forward and backward including cross-correlation
- GEMM (general matrix multiply)
- Pooling forward and backward
- Softmax forward and backward
- Neuron activations forward and backward: relu, tanh, sigmoid, elu, gelu, softplus, swishArithmetic, mathematical, relational, and logical pointwise operations (including various flavors of forward and backward neuron activations)
- Tensor transformation functions (reshape, transpose, concat, reshape, etc)
- LRN, LCN, batch normalization, instance normalization, and layer normalization forward and backward

Beyond just providing performant implementations of individual operations, the library also supports a flexible set of multi-operation fusion patterns for further optimization. The goal is to achieve the best available performance on NVIDIA GPUs for important deep learning use cases.

In cuDNN version 7 and older, the API was designed to support a fixed set of operations and fusion patterns. We informally call this the “legacy API”. Starting in cuDNN version 8, to address the quickly expanding set of popular fusion patterns, we added a [Graph API](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html#graph-api), which allows the user to express a computation by defining an operation graph, rather than by selecting from a fixed set of API calls. This offers better flexibility versus the legacy API, and for most use cases, is the recommended way to use cuDNN.

You may have initially confused the term “Graph API” with operations to do with graph neural networks. It turns out this just lets you define the graph of operations you’d prefer in the form of a Graph. Rather than using fixed operations (legacy API) you can’t actually see code for under the hood (since its a precompiled binary), you get an API you can add to without directly changing the low level source code. 

here is the rough idea when it comes to cuDNN docs:

you have these tensor descriptor types implemented as “opaque struct types” we previously talked about. these descriptors can create tensors, define tensor operations, get attributes about tensors, and more. 

we are going to reverse engineer the following code snippet ( you can type these into google search, find the graph API, and paste the cudnnConvolutionForward to find where the docs for this exist, then map out everything around it and dig into the descriptor types a little more

`cudnnTensorDescriptor_t`

`cudnnHandle_t`

`cudnnConvolutionDescriptor_t`

`cudnnFilterDescriptor_t`

`cudnnCreateTensorDescriptor`
`cudnnSetTensor4dDescriptor`

`cudnnConvolutionFwdAlgo_t`

`cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_kernel, convDesc, algo, workspace, workspaceSize, &beta, outputDesc, d_output_cudnn)`

we have a cudnn handle, a pointer to the alpha parameter (not descriptor type), input descriptor, the conv input on device memory, the conv filter/kernel descriptor, the kernel tensor itself, the conv operation descriptor, algo as the forward algorithm type (very top item after clicking on ⇒ https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-ops-library.html#id172), the memory workspace the GPU needs to do a conv operation (workspace & workspaceSize), beta is a pointer to a float param, output descriptor, output tensor on device memory.

you want cudnn to take in your input tensor as

```python
tensor([[[-1.7182,  1.2014, -0.0144],
         [-0.6332, -0.5842, -0.7202]],

        [[ 0.6992, -0.9595,  0.1304],
         [-0.0369,  0.8105,  0.8588]],

        [[-1.0553,  1.9859,  0.9880],
         [ 0.6508,  1.4037,  0.0909]],

        [[-0.6083,  0.4942,  1.9186],
         [-0.7630, -0.8169,  0.6805]]])
```

as a pytorch reference. but want you allocate memory its just an array of int/floats. 

```python
[-1.7182,  1.2014, -0.0144, -0.6332, -0.5842, -0.7202,  0.6992, -0.9595,
	0.1304, -0.0369,  0.8105,  0.8588, -1.0553,  1.9859,  0.9880,  0.6508,
	1.4037,  0.0909, -0.6083,  0.4942,  1.9186, -0.7630, -0.8169,  0.6805])
```

it turns out this part isn’t as bad as you would expect. notice how we have the shape (4, 2, 3). we can split into 4 equal sections (our batch elements), split each of those into 2 sections (maybe time dimension), at this point we are left with the original intended shape. this is how cudnn handles your tensors under the hood. as long as you specify the shape properly (ex: NCHW ⇒ batch_size, channels, height, width) you have nothing to worry about (still cudnn error check of course)

all code I used here is in `01 Conv2d.cu`


1. **Pre-compiled Single Operation Engines**:
    - These engines are pre-compiled and optimized for a specific single operation. Because they are pre-compiled, they offer very efficient execution but are inflexible in terms of the operations they can perform.
    - Example: A matrix multiplication engine that is pre-compiled and optimized specifically for that operation.
2. **Generic Runtime Fusion Engines**:
    - These engines are designed to dynamically fuse multiple operations at runtime. They offer more flexibility compared to pre-compiled engines since they can adapt to different combinations of operations but might not be as highly optimized as pre-compiled or specialized runtime engines.
    - Example: An engine that can dynamically fuse different element-wise operations on tensors during execution to avoid redundant memory reads/writes. (you can fuse uncommon operations together, gaining a decent improvement, but still not as fast as pre-compiled).
3. **Specialized Runtime Fusion Engines**:
    - Similar to generic runtime fusion engines, but these are specifically optimized for certain patterns or combinations of operations. They still offer runtime flexibility but also try to leverage optimizations for particular use cases or operation sequences.
    - Example: An engine optimized for fusing convolutional layers followed by activation functions in neural networks. It will recognize your code architecture or some pattern during the CUDA script compilation and find the fused operations in the backend where you would get a speedup
4. **Specialized Pre-compiled Fusion Engines**:
    - These engines are pre-compiled and optimized for specific sequences of operations. They offer the same high performance as pre-compiled single operation engines but can handle sequences of operations rather than just single ones.
    - Example: A pre-compiled engine for a specific convolutional block in a neural network that combines convolution, batch normalization, and ReLU activation functions.

### Runtime Fusion:

Consider a scenario where you need to perform several element-wise operations on tensors, such as addition, multiplication, and a sigmoid activation function. Without runtime fusion, each operation would be a separate kernel launch, each reading from and writing to global memory:

`output = torch.sigmoid(tensor1 + tensor2 * tensor3)`

With runtime fusion, the above operations could be combined into a single kernel launch, thus performing the entire computation in one go, keeping intermediate results in registers and only writing the final output to global memory.

## Graph API

- https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html
- Of course, for fusion to be interesting, the graph needs to support multiple operations. And ideally, we want the supported patterns to be flexible to cover a diverse set of use cases. To accomplish this generality, cuDNN has runtime fusion engines that generate the kernel (or kernels) at runtime based on the graph pattern. This section outlines the patterns supported by these runtime fusion engines (that is, engines with `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION` behavioral note).


![](../assets/knlfusion1.png)
![](../assets/knlfusion2.png)

you will have to check the compute compatibility of your GPU to see which of these operations will fuse

1. Graph API -> Kernel Fusion where nodes are "operations" and edges are "tensors"
2. Ops API -> Single Operation Engine (softmax, batchnorm, dropout, etc)
3. CNN API -> Convolution and related operations (depended on by Graph API)
4. Adversarial API -> "Other" features and algos (RNNs, CTC loss, multihead attn, etc)

## Performance Benchmarking
- say you want to find the fastest cudnn convolution forward algorithm for your use case
you would look at the different algorithms from algorithm type (something like `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`)
and compare the performance of each algorithm. 
- sometimes, you can get better performance by writing your own kernel instead of relying on cuDNN.
- looking back at the cudnn graph API, you can implement your own "graph" of operations and fuse them together resulting in a speedup for a certain chunk of the fwd/bkwd pass.
- if you're not batch processing, you might get away with writing your own optimized custom kernel (production cases)

## Navigating the cuDNN API
- just ctrl+click or cmd+click on the function names to see the source code (ex: `cudnnConvolutionForward`)
```cpp
cudnnConvolutionForward(cudnnHandle_t handle,
                        const void *alpha,
                        const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const cudnnFilterDescriptor_t wDesc,
                        const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo,
                        void *workSpace,
                        size_t workSpaceSizeInBytes,
                        const void *beta,
                        const cudnnTensorDescriptor_t yDesc,
                        void *y);
```