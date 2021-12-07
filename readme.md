# vapo
Implementation of Visual Affordance-guided Policy Optimization

# Installation
- Setup a conda environment by running:
```
git clone https://github.com/mees/vapo.git
cd vapo/
conda create -n vapo
conda activate vapo
pip install -e .
```

Install pytorch and set the CUDA toolkit version to match the native CUDA version (in /usr/local/cuda/), since you must compile the hough voting code with corresponding CUDA compiler (nvcc is not provided with the conda cudatoolkit distribution). This can be checked with: nvcc --version. This code was tested on pytorch 1.8 and cuda 10.1

- Install the affordance model
```
git clone https://github.com/JessicaBorja/affordance.git
conda activate vapo
cd affordance/
pip install -e .
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

Go to the directory of the voting layer and run [setup.py](./affordance_model/hough_voting/setup.py). If you do not have sudo privileges, don't run `sudo make install` instead change the diretory in "include_dirs" to match where the eigen-git-mirror repo was downloaded, then run: 

```
conda activate vapo
cd /ROOT_DIR/affordance_model/hough_voting/
python setup.py install
```

# Running experiments
## Reinforcement Learning policy
hydra configuration for tabletop experiments can be found in [cfg_tabletop.yaml]("./config/cfg_tabletop.yaml")
Name gets generated automatically depending on wether it uses a dense or sparse reward, if it uses the detected target and if it uses the affordance mask in the observation.

**Example**:
Baseline model
` python ./scripts/train_tabletop.py model_name=baseline paths.parent_folder=parent_folder`

This model will get the name full_sparse given the configuration parameters. parent_folder should point to the parent directory where vapo and the VREnv are stored.

VAPO
`python ./scripts/train_tabletop.py paths.parent_folder=~/ model_name=full affordance.gripper_cam.densify_reward=True affordance.gripper_cam.use_distance=True affordance.gripper_cam.use=True`


# Testing experiments
For testing both the affordance model and reinforcement learning policy, the hydra configuration that was generated during training is loaded. This way the model gets loaded with the correct parameters.
