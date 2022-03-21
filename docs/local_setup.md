# Installation
- Setup a conda environment by running:

```
git clone https://github.com/mees/vapo.git
cd vapo/
conda create -n vapo python==3.8
conda activate vapo
```
- Install pytorch

To install the voting layer the cudatoolkit installed with pytorch must match the native CUDA version (in /usr/local/cuda/) which will be used to compile the CUDA code. Otherwise, the compiled CUDA/C++ code may not be compatible with the conda-installed PyTorch.

First check your CUDA version with nvcc --version or in /usr/local/cuda/version.json then install [pytorch](https://pytorch.org/get-started/locally/) with the corresponding toolkit version. This code was tested with pytorch 1.10 and cuda 11.3.

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

- Install the Hough voting layer

The hough voting layer implementation was taken from [uois2d repo](https://github.com/chrisdxie/uois/tree/uois2d). Please refer to their repository for more information about it. To install the voting layer first install [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page).
```
git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror/
mkdir build/
cd build/
cmake ..
sudo make install
```

Go to the directory of the voting layer and run [setup.py](./vapo/affordance/hough_voting/setup.py). If you do not have sudo privileges, don't run `sudo make install` instead change the diretory in "include_dirs" to match where the eigen-git-mirror repo was downloaded, then run: 

```
conda activate vapo
cd /VAPO_ROOT/vapo/affordance/hough_voting/
python setup.py install
```

- Install extra dependencies
```
cd /VAPO_ROOT/
pip install -e .
```

- Install the VRENv
[Setup](https://github.com/JessicaBorja/VREnv/blob/master/docs/setup.md)

# Setting up git branches
- Checkout to RLSupport branch on VRENV
- Checkout to dev_jessica on VRData

