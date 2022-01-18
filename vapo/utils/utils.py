import os
import gym
from omegaconf import OmegaConf
from vapo.sac_agent.sac_utils.utils import set_init_pos
from affordance.affordance_model import AffordanceModel
import hydra


def get_abs_path(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def torch_to_numpy(x):
    return x.detach().cpu().numpy()


def init_aff_net(affordance_cfg, cam_str=None, in_channels=1):
    aff_net = None
    if(affordance_cfg is not None):
        if(cam_str is not None):
            aff_cfg = affordance_cfg["%s_cam" % cam_str]
        else:
            aff_cfg = affordance_cfg
        if("use" in aff_cfg and aff_cfg.use):
            path = aff_cfg.model_path
            path = get_abs_path(path)
            # Configuration of the model
            hp = {**aff_cfg.hyperparameters,
                  "in_channels": in_channels}
            hp = OmegaConf.create(hp)
            # Create model
            if(os.path.exists(path)):
                aff_net = AffordanceModel.load_from_checkpoint(
                                    path,
                                    cfg=hp)
                aff_net.cuda()
                aff_net.eval()
                print("obs_wrapper: %s cam affordance model loaded" % cam_str)
            else:
                affordance_cfg = None
                raise TypeError("Path does not exist: %s" % path)
    return aff_net


def change_project_path(cfg, run_cfg):
    net_cfg = run_cfg.agent.net_cfg
    # Change affordance path to match current system
    static_cam_aff_path = net_cfg.affordance.static_cam.model_path
    static_cam_aff_path = static_cam_aff_path.replace(
        run_cfg.models_path,
        cfg.models_path)
    net_cfg.affordance.static_cam.model_path = static_cam_aff_path

    # Gripper cam
    gripper_cam_aff_path = net_cfg.affordance.gripper_cam.model_path
    gripper_cam_aff_path = gripper_cam_aff_path.replace(
        run_cfg.models_path,
        cfg.models_path)
    net_cfg.affordance.gripper_cam.model_path = gripper_cam_aff_path

    # Static cam target_search
    target_search = run_cfg.target_search.model_path
    target_search = target_search.replace(
        run_cfg.models_path,
        cfg.models_path)
    run_cfg.target_search.model_path = target_search

    # VREnv data path
    run_cfg.models_path = cfg.models_path
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
    if('rand_init_state' in run_cfg.env):
        run_cfg.env.pop('rand_init_state')
        run_cfg.eval_env.pop('rand_init_state')
    return run_cfg, net_cfg, env_wrapper, agent_cfg


def get_3D_end_points(x, y, z, w, h, d):
    w = w/2
    h = h/2
    box_top_left = [x - w, y + h, z]
    box_bott_right = [x + w, y - h, z + d]
    return (box_top_left, box_bott_right)


def register_env():
    gym.envs.register(
        id='VREnv-v0',
        entry_point='VREnv.vr_env.envs.play_table_env:PlayTableSimEnv',
        max_episode_steps=200,
    )
