from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='rle',
    description="a package used for compress sparse tensor",
    packages=["rle"],
    package_data={"rle": ["__init__.py"]},
    ext_modules=[
        cpp_extension.CUDAExtension('rle_cuda', [
            'rle_cuda.cpp',
            'rle_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })
