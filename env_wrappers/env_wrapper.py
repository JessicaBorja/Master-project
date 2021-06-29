from .observation_wrapper import ObservationWrapper
from .reward_wrapper import RewardWrapper
import os
import logging
import torch
import gym
from vr_env.envs.play_table_env import PlayTableSimEnv
from vr_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
logger = logging.getLogger(__name__)


def wrap_env(env, max_ts, save_images=False, **args):
    env = ObservationWrapper(env, save_images=save_images, **args)
    env = RewardWrapper(env, max_ts)
    return env


def init_env(env_cfg):
    if(env_cfg.use_egl or env_cfg.use_egl):
        device = torch.device(torch.cuda.current_device())
        set_egl_device(device)
    env = PlayTableSimEnv(**env_cfg)
    return env


def set_egl_device(device):
    assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
    cuda_id = device.index if device.type == "cuda" else 0
    try:
        egl_id = get_egl_device_id(cuda_id)
    except EglDeviceNotFoundError:
        logger.warning(
            "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
            "When using DDP with many GPUs this can lead to OOM errors. "
            "Did you install PyBullet correctly? Please refer to VREnv README"
        )
        egl_id = 0
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")