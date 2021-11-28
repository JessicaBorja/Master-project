from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# If you do not have sudo priviledges
# Change to include_dirs = '/directory/eigen-git-mirror/'

setup(
    name='hough_voting',
    ext_modules=[
        CUDAExtension(
            name='hough_voting_cuda',
            sources=['hough_voting_layer.cpp',
                     'hough_voting_kernel.cu'],
                     include_dirs=['/home/jessica/eigen-git-mirror/'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
