import sys
import os
import gym
from omegaconf import OmegaConf
from sac_agent.sac_utils.utils import set_init_pos


def change_project_path(cfg, run_cfg):
    net_cfg = run_cfg.agent.net_cfg
    # Change affordance path to match current system
    static_cam_aff_path = net_cfg.affordance.static_cam.model_path
    static_cam_aff_path = static_cam_aff_path.replace(
        run_cfg.project_path,
        cfg.project_path)
    net_cfg.affordance.static_cam.model_path = static_cam_aff_path

    # Gripper cam
    gripper_cam_aff_path = net_cfg.affordance.gripper_cam.model_path
    gripper_cam_aff_path = gripper_cam_aff_path.replace(
        run_cfg.project_path,
        cfg.project_path)
    net_cfg.affordance.gripper_cam.model_path = gripper_cam_aff_path

    # Static cam target_search
    target_search_aff = run_cfg.target_search_aff.model_path
    target_search_aff = target_search_aff.replace(
        run_cfg.project_path,
        cfg.project_path)
    run_cfg.target_search_aff.model_path = target_search_aff

    # VREnv data path
    run_cfg.data_path = run_cfg.data_path.replace(
        run_cfg.project_path,
        cfg.project_path)


def load_cfg(cfg_path, cfg, optim_res):
    if(os.path.exists(cfg_path) and not optim_res):
        run_cfg = OmegaConf.load(cfg_path)
        net_cfg = run_cfg.agent.net_cfg
        env_wrapper = run_cfg.env_wrapper
        agent_cfg = run_cfg.agent.hyperparameters
        change_project_path(cfg, run_cfg)
    else:
        run_cfg = cfg
        net_cfg = cfg.agent.net_cfg
        env_wrapper = cfg.env_wrapper
        agent_cfg = cfg.agent.hyperparameters

    if('init_pos_near' in run_cfg):
        if(run_cfg.init_pos_near):
            init_pos = run_cfg.env.robot_cfg.initial_joint_positions
            init_pos = set_init_pos(run_cfg.task, init_pos)
            run_cfg.env.robot_cfg.initial_joint_positions = init_pos
            run_cfg.eval_env.robot_cfg.initial_joint_positions = init_pos
    return run_cfg, net_cfg, env_wrapper, agent_cfg


def register_env():
    insert_path()
    gym.envs.register(
        id='VREnv-v0',
        entry_point='VREnv.vr_env.envs.play_table_env:PlayTableSimEnv',
        max_episode_steps=200,
    )


def insert_path():
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, parent_dir+"/VREnv/")
