import gym
import cv2
import numpy as np
from omegaconf.omegaconf import OmegaConf
import torch
import os
import pybullet as p

from vapo.affordance_model.segmentator_centers import Segmentator
from vapo.affordance_model.datasets import get_transforms
import vapo.utils.flowlib as flowlib
from vapo.utils.img_utils import overlay_mask, overlay_flow
import matplotlib.pyplot as plt
from omegaconf.listconfig import ListConfig

def get_name(cfg, model_name):
    if(cfg.env_wrapper.gripper_cam.use_img):
        model_name += "_img"
    if(cfg.env_wrapper.gripper_cam.use_depth):
        model_name += "_depth"
    if(cfg.affordance.gripper_cam.target_in_obs):
        model_name += "_target"
    if(cfg.affordance.gripper_cam.use_distance):
        model_name += "_dist"
    if(cfg.affordance.gripper_cam.use):
        model_name += "_affMask"
    if(cfg.affordance.gripper_cam.densify_reward):
        model_name += "_dense"
    else:
        model_name += "_sparse"
    return model_name


def find_cam_ids(cameras):
    static_id, gripper_id, render_cam_id = 0, 1, None
    for i, cam in enumerate(cameras):
        if "gripper" in cam.name:
            gripper_id = i
        elif "static" in cam.name:
            static_id = i
        elif "render" in cam.name:
            render_cam_id = i
    cameras = {"static": static_id,
               "gripper": gripper_id}
    if(render_cam_id):
        cameras.update({"rendering": render_cam_id})
    return cameras


def get_obs_space(affordance_cfg, gripper_cam_cfg, static_cam_cfg,
                  channels, img_size, use_robot_obs, task, oracle=False):
    cfg_dict = {
        "gripper": [gripper_cam_cfg, affordance_cfg.gripper_cam],
        "static": [static_cam_cfg, affordance_cfg.static_cam]
    }
    obs_space_dict = {}
    for cam_type, config in cfg_dict.items():
        env_obs_cfg, aff_cfg = config
        if(env_obs_cfg.use_img):
            obs_space_dict["%s_img_obs" % cam_type] = gym.spaces.Box(
                low=0, high=255,
                shape=(channels, img_size, img_size))
        if(env_obs_cfg.use_depth):
            obs_space_dict['%s_depth_obs' % cam_type] = gym.spaces.Box(
                low=0, high=255,
                shape=(1, img_size, img_size))
        if(aff_cfg.use):
            obs_space_dict['%s_aff' % cam_type] = gym.spaces.Box(
                low=0, high=1,
                shape=(1, img_size, img_size))
    if(use_robot_obs):
        # *tcp_pos(3), *tcp_euler(1), gripper_width, gripper_action(1),
        if(task == "pickup"):
            obs_space_dict['robot_obs'] = gym.spaces.Box(
                low=-0.5, high=0.5, shape=(6,))
        else:
            obs_shape = 9 if oracle else 8
            obs_space_dict['robot_obs'] = gym.spaces.Box(
                low=-0.5, high=0.5, shape=(obs_shape,))
    if(affordance_cfg.gripper_cam.target_in_obs):
        obs_space_dict['detected_target_pos'] = gym.spaces.Box(
            low=-1, high=1, shape=(3,))
    if(affordance_cfg.gripper_cam.use_distance):
        obs_space_dict['target_distance'] = gym.spaces.Box(
            low=-1, high=1, shape=(1,))
    return gym.spaces.Dict(obs_space_dict)


def init_aff_net(affordance_cfg, cam_str, in_channels=1):
    aff_net = None
    if(affordance_cfg):
        if(affordance_cfg.static_cam.use
           and cam_str == "static"):
            path = affordance_cfg.static_cam.model_path
            # Configuration of the model
            hp = {**affordance_cfg.static_cam.hyperparameters,
                  "hough_voting": affordance_cfg.static_cam.hough_voting,
                  "in_channels": in_channels}
            hp = OmegaConf.create(hp)

            # Create model
            if(os.path.exists(path)):
                aff_net = Segmentator.load_from_checkpoint(
                                    path,
                                    cfg=hp)
                aff_net.cuda()
                aff_net.eval()
                print("obs_wrapper: Static cam affordance model loaded")
            else:
                affordance_cfg = None
                path = os.path.abspath(path)
                raise TypeError("Path does not exist: %s" % path)
        elif(cam_str == "gripper" and
                (affordance_cfg.gripper_cam.use or
                 affordance_cfg.gripper_cam.densify_reward)):
            path = affordance_cfg.gripper_cam.model_path

            # Configuration of the model
            hp = {**affordance_cfg.gripper_cam.hyperparameters,
                  "hough_voting": affordance_cfg.gripper_cam.hough_voting}
            hp = OmegaConf.create(hp)

            # Create model
            if(os.path.exists(path)):
                aff_net = Segmentator.load_from_checkpoint(path, cfg=hp)
                aff_net.cuda()
                aff_net.eval()
                print("obs_wrapper: gripper affordance model loaded")
            else:
                affordance_cfg = None
                path = os.path.abspath(path)
                raise TypeError("Path does not exist: %s" % path)
    return aff_net, affordance_cfg


def depth_preprocessing(frame, img_size):
    # obs is from 0-255, (img_size, img_size, 1)
    new_frame = cv2.resize(
            frame,
            (img_size, img_size),
            interpolation=cv2.INTER_AREA)
    # (1, img_size, img_size)
    new_frame = np.expand_dims(new_frame, axis=0)
    return new_frame


def get_transforms_and_shape(transforms_cfg, in_size, out_size=None):
    apply_transforms = get_transforms(transforms_cfg, img_size=out_size)
    test_tensor = torch.zeros((3, in_size, in_size))
    test_tensor = apply_transforms(test_tensor)
    # History length
    if(test_tensor.shape == 2):
        test_tensor = test_tensor.unsqueeze(0)
    return apply_transforms, test_tensor.shape


def load_aff_from_hydra(cfg):
    # Initialize model
    hydra_cfg_path = cfg.folder_name + "/.hydra/config.yaml"
    if os.path.exists(hydra_cfg_path):
        run_cfg = OmegaConf.load(hydra_cfg_path)
    else:
        print("path does not exist %s" % hydra_cfg_path)
        run_cfg = cfg
    model_cfg = run_cfg.model_cfg
    model_cfg.hough_voting = cfg.model_cfg.hough_voting

    # Load model
    checkpoint_path = os.path.join(cfg.folder_name, "trained_models")
    checkpoint_path = os.path.join(checkpoint_path, cfg.model_name)
    model = Segmentator.load_from_checkpoint(checkpoint_path,
                                             cfg=model_cfg).cuda()
    model.eval()
    print("model loaded")
    return model, run_cfg


def img_preprocessing(frame, transforms):
    # obs is from 0-255, (img_size, img_size, 3)
    frame = torch.from_numpy(frame).permute(2, 0, 1)
    frame = transforms(frame)
    frame = frame.cpu().detach().numpy()

    # History length
    # if(frame.shape == 2):
    #     frame = np.expand_dims(frame, 0)  # (1, img_size, img_size)
    # c, w, h = frame.shape
    # self._total_frames = np.pad(self._total_frames, ((
    #     c, 0), (0, 0), (0, 0)), mode='constant')[:-c, :, :]
    # self._total_frames[0:c] = frame

    # self._cur_img_obs = [self._total_frames[i * c:(i * c) + c]
    #                      for i in self._indices]
    # self._cur_img_obs = np.concatenate(self._cur_img_obs, 0)
    return frame

def transform_and_predict(model, img_transforms, orig_img,
                          img_resize, show=True, rgb=False,
                          cam="gripper",
                          out_shape=(200, 200)):
    resize_shape = (img_resize, img_resize)
    if(rgb):
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    orig_img_resize = cv2.resize(orig_img, resize_shape)
    # Apply validation transforms
    x = torch.from_numpy(orig_img_resize).permute(2, 0, 1).unsqueeze(0)
    if isinstance(img_transforms, ListConfig):
        img_transforms = get_transforms(img_transforms, img_resize)
    x = img_transforms(x).cuda()

    # Predict affordance, centers and directions
    aff_logits, _, aff_mask, directions = model(x)
    fg_mask, _, object_centers, object_masks = \
        model.predict(aff_mask, directions)

    # To numpy arrays
    pred_shape = np.array(fg_mask[0].shape)
    mask = fg_mask.detach().cpu().numpy()
    object_masks = object_masks.permute((1, 2, 0)).detach().cpu().numpy()
    directions = directions[0].detach().cpu().numpy()

    # Plot different objects according to voting layer
    obj_segmentation = orig_img_resize
    centers = []
    obj_class = np.unique(object_masks)[1:]
    obj_class = obj_class[obj_class != 0]  # remove background class
    cm = plt.get_cmap('tab10')
    colors = cm(np.linspace(0, 1, len(obj_class)))[:, :3]
    colors = (colors * 255).astype('uint8')
    for i, o in enumerate(object_centers):
        o = o.detach().cpu().numpy()
        centers.append(o)

    # Affordance segmentation
    affordances = orig_img
    bin_mask = (mask[0] * 255).astype('uint8')
    bin_mask = cv2.resize(bin_mask, affordances.shape[:2])
    affordances = overlay_mask(bin_mask, affordances, (255, 0, 0))

    # To flow img
    directions = np.transpose(directions, (1, 2, 0))
    flow_img = flowlib.flow_to_image(directions)  # RGB
    flow_img = flow_img[:, :, ::-1]  # BGR
    mask = (np.transpose(mask, (1, 2, 0))*255).astype('uint8')
    # mask[:, 100:] = 0

    # Resize to out_shape
    obj_segmentation = cv2.resize(obj_segmentation, out_shape)
    orig_img = cv2.resize(orig_img, out_shape)
    flow_img = cv2.resize(flow_img, out_shape)
    mask = cv2.resize(mask, out_shape)
    affordances = cv2.resize(affordances, out_shape)

    # Resize centers
    new_shape = np.array(out_shape)
    for i in range(len(centers)):
        centers[i] = (centers[i] * new_shape / pred_shape).astype("int32")

    # Overlay directions and centers
    res = overlay_flow(flow_img, orig_img, mask)

    # Draw detected centers
    for c in centers:
        u, v = c[1], c[0]  # center stored in matrix convention
        res = cv2.drawMarker(res, (u, v),
                                (0, 0, 0),
                                markerType=cv2.MARKER_CROSS,
                                markerSize=10,
                                line_type=cv2.LINE_AA)
    if(show):
        # cv2.imshow("Affordance masks", affordances)
        cv2.imshow("object masks: % s" % cam, obj_segmentation)
        cv2.imshow("%s output" % cam, res)
        # cv2.imshow("flow", flow_img)
        # cv2.waitKey(1)