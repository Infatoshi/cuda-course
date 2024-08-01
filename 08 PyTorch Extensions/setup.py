from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='polynomial_cpp',
    ext_modules=[
        CUDAExtension('polynomial_cpp', [
            'polynomial.cpp',
            'polynomial.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })