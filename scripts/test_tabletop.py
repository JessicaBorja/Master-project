import hydra
import os

from vapo.env_wrappers.play_table_rl import PlayTableRL
from vapo.env_wrappers.aff_wrapper import AffordanceWrapper
from vapo.combined.combined import Combined
from vapo.utils.utils import load_cfg


@hydra.main(config_path="../config", config_name="cfg_tabletop")
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
    run_cfg.target_search_aff = cfg.target_search_aff

    max_ts = cfg.agent.learn_config.max_episode_length

    save_images = cfg.test.eval_cfg.save_images
    env = AffordanceWrapper(PlayTableRL, run_cfg.eval_env, max_ts,
                            affordance_cfg=run_cfg.affordance,
                            viz=cfg.viz_obs,
                            save_images=save_images,
                            **run_cfg.env_wrapper)

    sac_cfg = {"env": env,
               "model_name": run_cfg.model_name,
               "save_dir": run_cfg.agent.save_dir,
               "net_cfg": net_cfg,
               **agent_cfg}

    model = Combined(run_cfg,
                     sac_cfg=sac_cfg,
                     target_search_mode="affordance")
    original_dir = hydra.utils.get_original_cwd()
    model_path = os.path.join(original_dir, cfg.resume_model_path)
    path = "%s/trained_models/%s.pth" % (cfg.test.folder_name,
                                         cfg.test.model_name)
    success = model.load(path, load_replay_buffer=False)
    if(success):
        model.tidy_up(env)
        # model.evaluate(env)
        # model.eval_all_objs(env,
        #                     **cfg.test.eval_cfg)

    env.close()


if __name__ == "__main__":
    main()
