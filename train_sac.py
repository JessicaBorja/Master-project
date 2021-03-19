import gym
import hydra
from omegaconf import OmegaConf
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.vr_env.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)
from utils.env_processing_wrapper import EnvWrapper
from sac_agent.sac import SAC


def set_init_pos(task, init_pos):
    if(task == "slide"):
        init_pos = [-1.7245799162898525, 1.8325999998998153, 1.0841486101692206, -2.4397926771547223, -0.6158132412445191, 3.8223000000000007, -0.7371749302934671]
    elif(task == "drawer"):
        init_pos = [-0.6674729645820069, 1.7509613833040443, 1.304967922545566, -2.3491822541374074, -1.8854745478807724, 1.6293532800388535, -0.6413070674224167]
    elif(task == "banana"):
        init_pos = [-0.9362933014487528, 1.7627833953065635, 1.2982581683461047, -2.155751832748698, -1.9943154775966752, 2.161505259849768, -1.2318332432817771]
    return init_pos

@hydra.main(config_path="./config", config_name="cfg_sac")
def main(cfg):
    print("agent configuration")
    print(OmegaConf.to_yaml(cfg.agent))
    print("repeat_training:%d" % cfg.repeat_training)
    for k, v in cfg.env_wrapper.items():
        print("%s:%s" % (k, str(v)))

    init_pos = cfg.env.robot_cfg.initial_joint_positions
    init_pos = set_init_pos(cfg.task, init_pos)
    cfg.env.robot_cfg.initial_joint_positions = init_pos
    cfg.eval_env.robot_cfg.initial_joint_positions = init_pos

    for i in range(cfg.repeat_training):
        training_env = gym.make("VREnv-v0", **cfg.env).env
        eval_env = gym.make("VREnv-v0", **cfg.eval_env).env
        training_env = EnvWrapper(training_env, train=True, **cfg.env_wrapper)
        eval_env = EnvWrapper(eval_env, **cfg.env_wrapper)
        model_name = cfg.model_name
        model = SAC(env=training_env,
                    eval_env=eval_env,
                    model_name=model_name,
                    save_dir=cfg.agent.save_dir,
                    net_cfg=cfg.agent.net_cfg,
                    **cfg.agent.hyperparameters)

        model.learn(**cfg.agent.learn_config)
        training_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
