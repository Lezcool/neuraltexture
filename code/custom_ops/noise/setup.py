from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
os.environ["CXX"] = "/usr/bin/g++-10"
os.environ["CC"] = "/usr/bin/gcc-10"
os.environ["CUDA_ROOT"] = "/usr/local/cuda"
setup(
    name='noise_cuda',
    ext_modules=[
        CUDAExtension('noise_cuda', [
            'noise_cuda.cpp',
            'noise_cuda_kernel.cu',
        ],
    #extra_compile_args=['-std=c++10']
    ),
    
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    )
