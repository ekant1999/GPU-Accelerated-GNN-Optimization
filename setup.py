import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Build CUDA extension only when CUDA is available
ext_modules = []
if os.environ.get('CUDA_HOME') or (os.environ.get('FORCE_CUDA', '').lower() == '1'):
    ext_modules.append(
        CUDAExtension(
            name='gnn_custom_ops',
            sources=[
                'csrc/spmm/spmm_kernel.cu',
                'csrc/spmm/spmm_csr.cu',
                'csrc/fused/fused_message_passing.cu',
                'csrc/reduction/scatter_reduce.cu',
                'csrc/bindings.cpp',
            ],
            include_dirs=['csrc'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', '--use_fast_math', '-std=c++17',
                    '-gencode=arch=compute_70,code=sm_70',   # V100
                    '-gencode=arch=compute_80,code=sm_80',   # A100
                    '-gencode=arch=compute_86,code=sm_86',   # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',   # RTX 4090
                ],
            },
        ),
    )
    cmdclass = {'build_ext': BuildExtension}
else:
    cmdclass = {}

setup(
    name='gnn_opt',
    packages=find_packages(exclude=('tests', 'scripts', 'configs', 'results', 'csrc')),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'torch>=2.0.0',
        'torch-geometric>=2.4.0',
        'pyyaml',
        'numpy',
        'scipy',
        'ogb>=1.3.6',
    ],
)
