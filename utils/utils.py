import sys
import os
import gym


def register_env():
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, parent_dir+"/VREnv/")
    gym.envs.register(
        id='VREnv-v0',
        entry_point='VREnv.vr_env.envs.play_table_env:PlayTableSimEnv',
        max_episode_steps=200,
    )
