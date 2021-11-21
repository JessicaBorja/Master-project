from setuptools import setup

setup(name='vapo',
      version='1.0',
      description='Python Distribution Utilities',
      packages=['vapo'],
      install_requires=[
          'hydra-core(==1.1.1)',
          'opencv-python(==4.5.3.56)',
          'pybullet(==3.1.7)',
          'pytorch-lightning',
          'segmentation-models-pytorch',
          'hydra-colorlog',
          'pypng',
          'tqdm',
          'omegaconf',
          'matplotlib']
     )
