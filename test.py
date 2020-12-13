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
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.src.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)

def evaluate(eval_config, model_name):
    env_name = "Pusher-v2"
    env = gym.make(env_name).env

    model = SAC(env, hidden_dim=470)
    success = model.load("./trained_models/%s.pth"%model_name)
    if(success):
        model_name = model_name.split("_r-")[0]#take model name, remove reward
        eval_config["model_name"] = model_name
        model.evaluate(env, **eval_config)

def load_config(model_name, folder_name):
    path = "./hydra_outputs/%s/.hydra/config.yaml"%(folder_name)
    path = os.path.abspath(path)
    cfg = yaml.load(open(path,'r'), Loader=yaml.FullLoader)
    eval_env_cfg = cfg["eval_env"]
    agent_cfg = cfg["agent"]["hyperparameters"]
    eval_env_cfg["cameras"].pop()
    eval_env_cfg["cameras"] = [cfg["static_camera"], cfg["gripper_camera"]]
    return eval_env_cfg, agent_cfg

def evaluateVRenv(eval_config, model_name, hydra_folderpath):
    evalenv_cfg, agent_cfg = load_config(model_name, hydra_folderpath)
    if(eval_config["render"]):
        evalenv_cfg["show_gui"]= True
    eval_env =  gym.make("VREnv-v0", **evalenv_cfg).env
    path = "./hydra_outputs/%s/trained_models/%s.pth"%(folder_name, model_name)
    agent_cfg["hidden_dim"] = 470
    print(agent_cfg)
    model = SAC(eval_env, **agent_cfg)
    success = model.load(path)
    if(success):
        eval_config["model_name"] = model_name
        model.evaluate(eval_env, **eval_config)

@hydra.main(config_path="./config", config_name="config_vrenv")
def hydra_evaluateVRenv(cfg):
    #Hinge
    if(cfg.task == "hinge"):
        model_name = "optim_hinge_rn1_rs1_04-12_03-17_best_eval"
        folder_name = "hinge/2020-12-04/12-44-23"
    elif(cfg.task == "drawer"):
        model_name = "optim_drawer_rn1_rs1_05-12_03-43_best_eval"
        folder_name = "drawer/2020-12-04/22-36-55"
        #img
        model_name = "drawer_img_default_13-12_01-42_best_eval"
        folder_name = "drawer/cluster/2020-12-13/01-42-26"
    else: #task == slide
        model_name = "optim_slide_rn1_rs1_05-12_04-28_best_eval"
        folder_name = "slide/2020-12-05/11-09-25"
        # #img
        # model_name = "slide_img_14-12_07-41_best_eval"
        # folder_name = "slide/cluster/2020-12-14/19-41-17"

    agent_cfg = cfg.agent.hyperparameters
    eval_config =  cfg.eval_config
    eval_env =  gym.make("VREnv-v0", **cfg.eval_env).env
    if(cfg.img_obs):
        eval_env =  ImgWrapper(eval_env)
    path = "../../../outputs/%s/trained_models/%s.pth"%(folder_name, model_name)
    print(os.path.abspath(path))
    print(agent_cfg)
    model = SAC(eval_env, img_obs=cfg.img_obs, **agent_cfg)
    success = model.load(path)
    if(success):
        model.evaluate(eval_env, model_name = model_name, **eval_config)

if __name__ == "__main__":
    #read_optim_results("sac_mujocoPusher2.pkl")
    #evaluateVRenv(eval_config, model_name, folder_name)
    hydra_evaluateVRenv()