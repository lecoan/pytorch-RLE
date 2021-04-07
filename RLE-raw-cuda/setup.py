from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rle',
    description="a package used for compress sparse tensor",
    packages=["rle"],
    package_data={"rle": "rle.py"},
    ext_modules=[
        CUDAExtension('rle_cuda', [
            'rle_cuda.cpp',
            'rle_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
