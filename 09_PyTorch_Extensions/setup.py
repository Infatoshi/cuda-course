from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='polynomial_cuda',
    ext_modules=[
        CUDAExtension('polynomial_cuda', [
            'polynomial_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })