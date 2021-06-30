import gym
import hydra
from omegaconf import OmegaConf
from utils.utils import register_env
from env_wrappers.env_wrapper import ObservationWrapper
from sac_agent.sac import SAC
from sac_agent.sac_utils.utils import set_init_pos
register_env()


def print_common_cfg(cfg):
    print("Learn config")
    print(OmegaConf.to_yaml(cfg.agent.learn_config))
    print("Image Wrapper:")
    for k, v in cfg.env_wrapper.items():
        print("%s:%s" % (k, str(v)))
    print("Affordance model:")
    for k, v in cfg.agent.net_cfg.affordance.items():
        print("%s:%s" % (k, str(v)))
    print("Train robot init: %s" %
          cfg.env.robot_cfg.initial_joint_positions)
    print("Eval robot init: %s" %
          cfg.eval_env.robot_cfg.initial_joint_positions)
    print("task: %s" % cfg.task)
    print("init_pos_near: %s" % cfg.init_pos_near)
    print("repeat_training:%d" % cfg.repeat_training)
    print("Sparse reward: %s" % cfg.sparse_reward)
    print("rand_init_state: %s" % cfg.rand_init_state)


@hydra.main(config_path="./config", config_name="cfg_sac")
def main(cfg):
    if(cfg.init_pos_near):
        init_pos = cfg.env.robot_cfg.initial_joint_positions
        init_pos = set_init_pos(cfg.task, init_pos)
        cfg.env.robot_cfg.initial_joint_positions = init_pos
        cfg.eval_env.robot_cfg.initial_joint_positions = init_pos
    print_common_cfg(cfg)

    for i in range(cfg.repeat_training):
        training_env = gym.make("VREnv-v0", **cfg.env).env
        eval_env = gym.make("VREnv-v0", **cfg.eval_env).env
        training_env = ObservationWrapper(training_env, train=True,
                                          affordance=cfg.affordance,
                                          **cfg.env_wrapper)
        eval_env = ObservationWrapper(eval_env,
                                      affordance=cfg.affordance,
                                      **cfg.env_wrapper)
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
