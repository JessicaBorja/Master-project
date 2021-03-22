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
        init_pos = [-1.1686195081948965, 1.5165126497924815, 1.7042540963745911, -1.6031852712241403, -2.5717679087567484, 2.331416872629473, -1.3006358472301627]
    elif(task == "drawer"):
        init_pos = [-0.4852725866746207, 1.0618989199760496, 1.3903811172536515, -1.7446581003391255, -1.1359501486104144, 1.8855365146855005, -1.3092771579652827]
    elif(task == "banana"):
        init_pos = [0.03740465833778156, 1.1844912206595481, 1.1330028132229706, -0.6702560563758552, -1.1188250499368455, 1.6153329476732947, -1.7078632665627795]
    elif(task == "hinge"):
        init_pos = [-0.3803066514807313, 0.931053115322005, 1.1668869976984892, -0.8602164833917604, -1.4818301463768684, 2.78299286093898, -1.7318962831826747]
    return init_pos

@hydra.main(config_path="./config", config_name="cfg_sac")
def main(cfg):
    print("agent configuration")
    print(OmegaConf.to_yaml(cfg.agent))
    print("repeat_training:%d" % cfg.repeat_training)
    print("Image Wrapper:")
    for k, v in cfg.env_wrapper.items():
        print("%s:%s" % (k, str(v)))
    print("Affordance model:")
    for k, v in cfg.agent.net_cfg.affordance.items():
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
