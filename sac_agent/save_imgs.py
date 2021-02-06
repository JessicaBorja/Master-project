import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import hydra
import pybullet as p
import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
#from gym.envs.mujoco import mujoco_env
#import mujoco_py
from sac import SAC
import os, sys, inspect
import yaml
from utils.env_img_wrapper import ImgWrapper
from omegaconf import DictConfig, OmegaConf
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.src.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)

def load_config(model_name, folder_name):
    path = "./hydra_outputs/%s/.hydra/config.yaml"%(folder_name)
    path = os.path.abspath(path)
    cfg = yaml.load(open(path,'r'), Loader=yaml.FullLoader)
    eval_env_cfg = cfg["eval_env"]
    agent_cfg = cfg["agent"]["hyperparameters"]
    eval_env_cfg["cameras"].pop()
    eval_env_cfg["cameras"] = [cfg["static_camera"], cfg["gripper_camera"]]
    return eval_env_cfg, agent_cfg


@hydra.main(config_path="./config", config_name="config_vrenv")
def save_imgs(cfg):
    #Hinge
    if(cfg.task == "hinge"):
        if(not cfg.img_obs):
            hidden_dim = 309
            model_name = "optim_hinge_rn1_rs1_04-12_03-17_best_eval"
            folder_name = "hinge/2020-12-04/12-44-23"
        else:
            #cluster\2020-12-27\01-35-14\results\slide_img_27-12_01-35 
            #me equivoque en el nombre de modelo y por eso dice slide pero es de hinge
            # hidden_dim = 488
            # model_name = "slide_img_27-12_01-35_best_eval"
            # folder_name = "hinge/cluster/2020-12-27/01-35-14"

            #cluster\2020-12-29\06-15-08\results\hinge_img_optim_31-12_01-02
            hidden_dim = 300 
            model_name = "hinge_img_optim_31-12_01-02_best_eval"
            folder_name = "hinge/cluster/2020-12-29/06-15-08"
    elif(cfg.task == "drawer"):
        if(not cfg.img_obs):
            hidden_dim=313
            model_name = "optim_drawer_rn1_rs1_05-12_03-43_best_eval"
            folder_name = "drawer/2020-12-04/22-36-55"
        else:
            #img
            # model_name = "drawer_img_default_13-12_01-42_best_eval"
            # folder_name = "drawer/cluster/2020-12-13/01-42-26"
            model_name = "drawer_x1img_pos_15-01_13-09_best_eval"
            folder_name = "drawer/cluster/2021-01-15/13-09-23"
    else: #task == slide
        if(not cfg.img_obs):
            hidden_dim=265
            model_name = "optim_slide_rn1_rs1_05-12_04-28_best_eval"
            folder_name = "slide/2020-12-05/11-09-25"
        else:
            # #img
            # cluster\2020-12-13\16-39-08\results\slide_img_13-12_04-39
            model_name = "slide_img_13-12_04-39_best_eval"
            folder_name = "slide/cluster/2020-12-13/16-39-08"

            #cluster\2020-12-29\06-14-54\results\slide_img_optim_30-12_17-43
            # hidden_dim=300
            # model_name = "slide_img_optim_30-12_17-43_best_eval"
            # folder_name = "slide/cluster/2020-12-29/06-14-54"
            model_name = "slide_img_only_10-01_23-06_best"
            folder_name = "slide/cluster/2021-01-10/23-04-58"
    
    test_model_dir = "../../../../outputs/%s/"%folder_name
    #run_cfg = OmegaConf.load(test_model_dir + ".hydra/config.yaml")
    #agent_cfg = run_cfg.agent.hyperparameters
    #cfg.img_wrapper = run_cfg.img_wrapper
    agent_cfg = cfg.agent.hyperparameters
    try:
        agent_cfg.net_cfg.hidden_dim =  hidden_dim
    except:
        #No different hidden_dim
        pass
    eval_config =  cfg.eval_config
    eval_env =  gym.make("VREnv-v0", **cfg.eval_env).env
    if(cfg.img_obs):
        eval_env =  ImgWrapper(eval_env, **cfg.img_wrapper)
    path = "../../../../outputs/%s/trained_models/%s.pth"%(folder_name, model_name)
    print(os.path.abspath(path))
    print(agent_cfg)
    model = SAC(eval_env, img_obs=cfg.img_obs,net_cfg=cfg.agent.net_cfg, **agent_cfg)
    success = model.load(path)

    if(success):
        eval_config.write_file=True
        model.evaluate(eval_env, model_name = model_name, **eval_config)

if __name__ == "__main__":
    save_imgs()