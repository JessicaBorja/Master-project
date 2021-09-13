import os.path

import hydra
import logging
from env_wrappers.panda_env_wrapper import PandaEnvWrapper
from env_wrappers.env_wrapper import RLWrapper
from env_wrappers.utils import get_name
from combined.combined_real import Combined


@hydra.main(config_path="./config", config_name="cfg_tabletop_real")
def main(cfg):
    # Auto generate names given dense, aff-mask, aff-target
    log = logging.getLogger(__name__)
    cfg.model_name = get_name(cfg, cfg.model_name)
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.robot_env, robot=robot)
    env = PandaEnvWrapper(**cfg.panda_env_wrapper, env=env)

    training_env = RLWrapper(env=env,
                             max_ts=cfg.agent.learn_config.max_episode_length,
                             train=True,
                             affordance_cfg=cfg.affordance,
                             viz=cfg.viz_obs,
                             **cfg.env_wrapper)


    sac_cfg = {"env": training_env,
                "eval_env": None,
                "model_name": cfg.model_name,
                "save_dir": cfg.agent.save_dir,
                "net_cfg": cfg.agent.net_cfg,
                "log": log,
                **cfg.agent.hyperparameters}

    log.info("model: %s" % cfg.model_name)
    model = Combined(cfg,
                     sac_cfg=sac_cfg,
                     target_search_mode=cfg.target_search,
                     rand_target=True)
    if cfg.resume_training:
        original_dir = hydra.utils.get_original_cwd()
        model_path = os.path.join(original_dir, cfg.resume_model_path)
        path = "%s/trained_models/%s.pth" % (model_path,
                                             cfg.model_name + "_last")
        if(os.path.exists(path)):
            model.load(path)
        else:
            print("Model path does not exist: %s \n Training from start" % os.path.abspath(path))
    model.learn(**cfg.agent.learn_config)
    training_env.close()
    # eval_env.close()


if __name__ == "__main__":
    main()
