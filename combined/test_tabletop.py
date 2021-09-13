import hydra
import logging
from env_wrappers.panda_env_wrapper import PandaEnvWrapper
from env_wrappers.env_wrapper import RLWrapper
from env_wrappers.utils import get_name
from combined.combined_real import Combined


@hydra.main(config_path="../config", config_name="cfg_tabletop_real")
def main(cfg):
    test_cfg = cfg.test
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
                     rand_target=False)
    path = "%s/trained_models/%s.pth" % (
            test_cfg.folder_name,
            test_cfg.model_name)
    success = model.load(path)

    if(success):
        model.evaluate(env, **cfg.test.eval_cfg)
    env.close()


if __name__ == "__main__":
    main()
