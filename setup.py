from setuptools import setup

setup(name='vapo',
      version='1.0',
      description='Python Distribution Utilities',
      packages=['vapo'],
      install_requires=[
          'hydra-core',
          'opencv-python',
          'pybullet',
          'numpy-quaternion',
          'hydra-colorlog',
          'pypng',
          'tqdm',
          'wandb',
          'omegaconf',
          'matplotlib']
      )
