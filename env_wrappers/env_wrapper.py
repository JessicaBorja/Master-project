from .observation_wrapper import ObservationWrapper
from .reward_wrapper import RewardWrapper
import os
import logging
import torch
import gym
from vr_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
logger = logging.getLogger(__name__)


def wrap_env(env, max_ts, save_images=False, **args):
    env = ObservationWrapper(env, save_images=save_images, **args)
    env = RewardWrapper(env, max_ts)
    return env


class EGLWrapper(gym.Wrapper):
    def __init__(self, env, device):
        self.set_egl_device(device)
        super(EGLWrapper, self).__init__(env)
        return env

    @staticmethod
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