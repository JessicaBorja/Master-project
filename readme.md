# Installation
- Setup a conda environment by running:
```
git clone https://github.com/JessicaBorja/Master-project.git
cd Master-project/
conda env create -f conda_env.yml
```

In conda_env.yml, we set cudatoolkit=10.1 by default. Set the CUDA toolkit version to match the native CUDA version (in /usr/local/cuda/), since you must compile the hough voting code with corresponding CUDA compiler (nvcc is not provided with the conda cudatoolkit distribution). This can be checked with: nvcc --version.

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
conda activate Master-Project
cd /ROOT_DIR/affordance_model/hough_voting/
python setup.py install
```

# Running experiments
## Reinforcement Learning policy
hydra configuration can be found in [cfg_combined.yaml]("./config/cfg_combined.yaml")
Name gets generated automatically depending on wether it uses a dense or sparse reward, if it uses the detected target and if it uses the affordance mask in the observation.

**Example**:
Baseline model
` python train_combined.py model_name=baseline`

This model will get the name full_target_dense given the configuration parameters.

`python train_combined.py project_path=~/ model_name=full affordance.gripper_cam.densify_reward=True affordance.gripper_cam.target_in_obs=True`

## Affordance model visualization
hydra configuration can be found in [cfg_affordance.yaml]("./config/cfg_affordance.yaml")

# Testing experiments
For testing both the affordance model and reinforcement learning policy, the hydra configuration that was generated during training is loaded. This way the model gets loaded with the correct parameters.

## Reinforcement Learning policy
Test configuration can be found in [test_combined.yaml]("./config/test/test_combined.yaml"). Default configuration is the baseline.

**Example**:
` python ./combined/test.py model_name=model_to_load folder_name=hydra_folder_of_model"`

## Affordance model visualization
If desired you can specify a different configuration for the center prediction by modifying the parameters in model_cfg.hough_voting.

Visualization configuration can be found in [viz_affordances.yaml]("./config/viz_affordances.yaml")

### Configuration parameters
- **project_folder**: absolute path directory pointing to location where the affordance_model directory is.
- **data_dir**: list of directories storing images which are desired to test the affordance model on. If relative paths are provided, they should be with respect to the location of this repository.
- **folder_name**: path of where the hydra output for a given model was stored.
- **model_name**: model_name.ckpt will be loaded according to this parameter. The saved models can be found inside the hydra output folder ("/folder_name/trained_models)
- **save_images**: If true it will save the images from the configured static camera to the **output_dir** directory
- **imshow**: If true it will show the images from the configured static camera
- **img_size**: It will transform the input images to this size before inputing to the network.
- **out_size**: It will resize the output of the network to this size.
- **output_dir**: output directory to store saved images.
- **model_cfg**: Used to define the voting layer parameters for the center detection. This is not used during training but only during inference, hence here you can move them around to find what works best.

**Example**:

` python ./affordance_model/viz.py model_name=other_model folder_name="./hydra_output_folder/"`
