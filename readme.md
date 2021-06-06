# Installation

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