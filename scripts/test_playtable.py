import hydra
import os
from vapo.wrappers.play_table_rl import PlayTableRL
from vapo.wrappers.aff_wrapper import AffordanceWrapper
from vapo.agent.vapo_agent import VAPOAgent
from vapo.utils.utils import load_cfg


@hydra.main(config_path="../config", config_name="cfg_playtable")
def main(cfg):
    test_cfg = cfg.test
    run_cfg, net_cfg, env_wrapper, agent_cfg =\
        load_cfg(os.path.join(test_cfg.folder_name, ".hydra/config.yaml"),
                 cfg, optim_res=False)

    run_cfg.eval_env.show_gui = cfg.env.show_gui
    run_cfg.eval_env.cameras = cfg.env.cameras
    run_cfg.eval_env.use_egl = cfg.env.use_egl
    run_cfg.scene = cfg.scene

    # new change
    run_cfg.robot.use_target_pose = False
    run_cfg.target_search = cfg.target_search

    max_ts = cfg.agent.learn_config.max_episode_length

    save_images = cfg.test.eval_cfg.save_images
    env = PlayTableRL(**run_cfg.eval_env)
    env = AffordanceWrapper(env, max_ts,
                            affordance_cfg=run_cfg.affordance,
                            viz=cfg.viz_obs,
                            save_images=save_images,
                            **run_cfg.env_wrapper)

    sac_cfg = {"env": env,
               "model_name": run_cfg.model_name,
               "save_dir": run_cfg.agent.save_dir,
               "net_cfg": net_cfg,
               **agent_cfg}

    model = VAPOAgent(run_cfg,
                      sac_cfg=sac_cfg,
                      target_search_mode=run_cfg.target_search)
    path = "%s/trained_models/%s.pth" % (
            test_cfg.folder_name,
            test_cfg.model_name)
    success = model.load(path)

    if(success):
        model.evaluate(env, **cfg.test.eval_cfg)
    env.close()


if __name__ == "__main__":
    main()
