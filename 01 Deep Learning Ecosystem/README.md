# Chapter 01: The Current Deep Learning Ecosystem


### **DISCLAIMER:** 

This part doesn’t go over anything highly technical with CUDA. Better to show you the ecosystem rather than enter technical details blindly. From my experience learning this stuff, having a decent enough understanding of the ecosystem will help you map out everything properly, and it provides that initial motivation to learn. 

As we mine further into detail, I encourage you to research and play around with what you find interesting (you will come across cool stuff in this section). If you just listen to someone talk about a subject for 20 hours, you limit your learning. Understanding breadth and depth of deep learning infrastructure is tough to navigate. Getting uncomfortable and breaking things is the best way to learn.


## Research
- PyTorch ([PyTorch - Fireship](https://www.youtube.com/watch?v=ORMx45xqWkA&t=0s&ab_channel=Fireship))
    - If you’re watching this I assume you have at least some basic knowledge of PyTorch. If not, I suggest watching the PyTorch video by [Daniel Bourke](https://www.youtube.com/watch?v=Z_ikDlimN6A)
    - Pytorch comes with nightly and stable versions ⇒ https://discuss.pytorch.org/t/pytorch-nightly-vs-stable/105633
        
        the nightly releases are more likely to be unstable but offer bleeding edge pytorch updates and universal framework optimizations
        
    - Users prefer pytorch more due to the usability w/ Huggingface
    - You will find pre-trained models on torchvision (`pip install torchvision`) and `torch.hub`. The pytorch ecosystem has developed a more decentralized but slightly harder to navigate approach to getting pretrained models. People will often release their models on github repos instead of pushing to a centralized database of models. Huggingface is most commonly used due to community efforts
    - Good ONNX support
- TensorFlow ([TensorFlow - Fireship](https://www.youtube.com/watch?v=i8NETqtGHms))
    - Well documented and lots of community support. Also most used deep learning framework
    - Comparatively the slowest DL framework
    - Created by Google (designed for TPUs) and general purpose ML (SVM, decision trees, etc).
    - Pre-trained models can be found directly on ⇒ https://www.tensorflow.org/resources/models-datasets
    - Good support for pre-trained models download in 1-3 lines of code.
    - Limited ONNX support (`tf2onnx`)
- Keras
    - Similar to `torch.nn` for TensorFlow, but higher-level.
    - Separate library but deeply integrated with TF, serving as its primary high-level API
    - Complete framework for building and training modules instead of just neural network modules
- JAX ([JAX - Fireship](https://www.youtube.com/watch?v=_0D5lXDjNpw))
    - JIT-compiled Autograd Xccelerated Linear Algebra
    - Docs here ⇒ https://jax.readthedocs.io/en/latest/
    - Feels like numpy
    - Reddit sentiment on JAX ⇒ https://www.reddit.com/r/MachineLearning/comments/1b08qv6/d_is_it_worth_switching_to_jax_from/
    - JAX and Tensorflow are both developed by Google
    - Uses XLA (xccelerated linear algebra) compiler
    - `tf2onnx` supported
- MLX
    - Developed by Apple for Apple Silicon
    - Open-source framework
    - Focuses on high-performance machine learning on Apple devices
    - Designed for both training and inference
    - Optimized for Apple's Metal GPU architecture
    - Allows for dynamic computation graphs
    - Suitable for research and development of new ML models
- PyTorch Lightning
    - https://www.reddit.com/r/deeplearning/comments/t31ppy/for_what_reason_do_you_or_dont_you_use_pytorch/
    - mostly the boilerplate code reduction and distributed scaling
    - `Trainer()` as opposed to training loop


## Production
- Inference-only
    - vLLM
        - ‣
    - TensorRT
        - integrates well with pytorch for inference
        - supports ONNX for loading models
        - highly optimized cuda kernels with the following in mind
            - benefits from sparsity
            - inference quantization
            - hardware architecture
            - memory access patterns across VRAM vs on-chip memory
        - short for tensor RunTime
        - developed, designed, and maintained by Nvidia
        - built specifically for LLM inference
        - uses some of the techniques we cover in this course, but abstracts them away for usability
        - seems that tensorRT requires Onnx look into this
        - review https://nvidia.github.io/TensorRT-LLM/ vaguely
        - follow links in order
            - https://nvidia.github.io/TensorRT-LLM/
            - https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
            - https://pytorch.org/TensorRT/getting_started/installation.html#installation
        - 
- Triton
    - Developed and maintained by OpenAI ⇒ https://openai.com/index/triton/
    - ‣
    - CUDA-like, but in python and gets rid of clutter around kernel development in regular CUDA C/C++. Also matches record performance on Matrix Multiplication
    - Get started ⇒ https://triton-lang.org/main/index.html
    - Write your first Triton kernel ⇒ https://triton-lang.org/main/getting-started/tutorials/index.html
    - Triton Inference Server
        - https://developer.nvidia.com/triton-inference-server
        - ‣
        - 
    - https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf is the original Triton paper
    - triton-viz is Triton’s main profiling and visualization toolkit
    - Python to finely control what happens on the GPU without worrying about the unexpected intricacies and complexities in C/C++.
        - Removes explicit memory management `cudaMalloc`, `cudaMemcpy`, `cudaFree`
        - No need for error checking / macros `CUDA_CHECK_ERROR`
        - Reduced complexity when grid / block / thread level indexing on kernel launch parameters
    
![](../05%20Writing%20your%20First%20Kernels/assets/triton1.png)
    
- torch.compile
    - Gets more attention than TorchScript and is typically better performance
    - Compiles a model down to a static representation so the dynamic graph component of pytorch doesn’t have to worry about things changing. Runs the model as an optimized binary instead of default out-of-the-box pytorch
    - https://discuss.pytorch.org/t/the-difference-between-torch-jit-script-and-torch-compile/188167
- TorchScript
    - Can be faster in scenarios, especially when deployed in C++
    - Performance gains can be specific to your neural net architecture
    - https://discuss.pytorch.org/t/the-difference-between-torch-jit-script-and-torch-compile/188167
- ONNX Runtime
    - https://youtu.be/M4o4YRVba4o
    - “**ONNX Runtime training** can accelerate the model training time on multi-node NVIDIA GPUs for transformer models with a one-line addition for existing PyTorch training scripts”
    - Developed and maintained by Microsoft
- Detectron2
    - Supports training and inference
    - Computer vision project started at Facebook (Meta)
    - Detection and segmentation algorithms

## Low-Level
- CUDA
    - Compute unified device architecture (CUDA) can be thought of as a programming language for nvidia gpus.
    - CUDA libs ⇒ cuDNN, cuBLAS, cutlass (fast linear algebra and DL algorithms). cuFFT for fast convolutions (FFTs are covered in the course)
    - writing the kernel yourself based on the hardware architecture (Nvidia still does this under the hood for by passing in special flags to the compiler)
- ROCm
    - CUDA equivalent for AMD GPUs
- OpenCL
    - Open Computing Language
    - CPUs, GPUs, digital signal processors, other hardware
    - since NVIDIA designed CUDA, it will outperform OpenCL on Nvidia tasks. If you are doing work with embedded systems (EE/CE), this is still worth learning.

## Inference for Edge Computing & Embedded Systems
    
- Edge Computing refers to low-latency and highly efficient local computing in the context of real-world distributed systems like fleets. Tesla FSD is a prime example of edge computing because it has a neural net running locally on the car. It also has to send data back to Tesla so they can improve their models. 

- CoreML
    - Primarily for deployment of pre-trained models on Apple devices
    - Optimized for on-device inference
    - Supports on-device training
    - Supports a wide range of model types (vision, natural language, speech, etc.)
    - Integrates well with Apple's ecosystem (iOS, macOS, watchOS, tvOS)
    - Focuses on privacy by keeping data on-device
    - Allows model conversion from other frameworks
    - Designed for app developers to easily incorporate ML into their apps
- PyTorch Mobile
- TensorFlow Lite

## Easy to Use
- FastAI
    - High-level API: Built on top of PyTorch, FastAI provides a more user-friendly interface for common deep learning tasks.
    - Rapid prototyping: Designed for quick implementation of state-of-the-art deep learning models.
    - Best practices: Incorporates many best practices and recent advances in deep learning by default.
    - Less code: Typically requires less code to implement complex models compared to raw PyTorch.
    - Transfer learning: Excellent support for transfer learning out of the box.
- ONNX
    - Open Neural Network eXchange
    - `torch.onnx.export(model, dummy_input, "resnet18.onnx")`
    
    ```python
    import tensorflow as tf
    import tf2onnx
    import onnx
    
    # Load your TensorFlow model
    tf_model = tf.keras.models.load_model('path/to/your/model.h5')
    
    # Convert the model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model)
    
    # Save the ONNX model
    onnx.save(onnx_model, 'path/to/save/model.onnx')
    ```
    
    ![Untitled](assets/onnx.png)
    
- wandb
    - Short for weights and biases
    - Easy to integrate with projects w/ a few lines of code
    - Team collaboration
    - Compare experiments w/ an intuitive UI
    
    ![Untitled](assets/wandb.png)
        
    
## Cloud Providers
- AWS
    - EC2 instances
    - Sagemaker (jupyter notebooks on a cluster, human data labelling/annotation, model training & deployment on AWS infrastructure)
- Google Cloud
    - Vertex AI
    - VM instances
- Microsoft Azure
    - Deep speed
- OpenAI
- VastAI
    - link picture of UI here
- Lambda Labs
    - Cheap datacenter GPUs
## Compilers
- XLA
    - A domain-specific compiler for linear algebra that optimizes TensorFlow computations
    - Provides a lower-level optimization and code generation backend for JAX
    - Performs whole-program optimization, seeing beyond individual operations to optimize across the entire computation graph
    - Enables efficient execution on various hardware (CPUs, GPUs, TPUs) by generating optimized machine code
    - Implements advanced optimizations like operation fusion, which combines multiple operations into a single, more efficient kernel
    - Allows JAX to achieve high performance without manually writing hardware-specific code
- LLVM
- MLIR
- NVCC
    - Nvidia CUDA Compiler
    - Works on everything in the CUDA toolkit
    
    ![Untitled](../10%20Extras/assets/nvcc.png)
        
## Misc
- Huggingface