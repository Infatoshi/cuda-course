# Custom PyTorch Extensions

```bash
python polynomial_activation.py
```

- you might get something similiar to this:

```bash
Using /home/elliot/.cache/torch_extensions/py311_cu121 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/elliot/.cache/torch_extensions/py311_cu121/polynomial_cuda/build.ninja...
/home/elliot/.pyenv/versions/3.11.7/lib/python3.11/site-packages/torch/utils/cpp_extension.py:1965: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
Building extension module polynomial_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module polynomial_cuda...
PyTorch built-in: 0.1055 ms
CUDA extension: 0.0233 ms
Triton: 0.3037 ms
```
- notice in the top line how this is saved to `/home/elliot/.cache/torch_extensions/py311_cu121` (you can remove stuff in the .cache directory if it gets flooded with binaries)


## Learning Resources
- https://github.com/pytorch/extension-cpp
- https://pytorch.org/tutorials/advanced/cpp_custom_ops.html
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- https://pytorch.org/docs/stable/notes/extending.html
