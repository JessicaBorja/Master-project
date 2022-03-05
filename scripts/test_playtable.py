import hydra
import os
from vapo.wrappers.play_table_rl import PlayTableRL
from vapo.wrappers.affordance.aff_wrapper_sim import AffordanceWrapperSim
from vapo.agent.vapo_agent import VAPOAgent
from vapo.utils.utils import load_cfg


@hydra.main(config_path="../config", config_name="cfg_playtable")
def main(cfg):
    # Load model cfg
    original_dir = hydra.utils.get_original_cwd()
    run_dir = os.path.join(original_dir, cfg.test.folder_name)
    run_dir = os.path.abspath(run_dir)
    run_cfg, net_cfg, env_wrapper, agent_cfg =\
        load_cfg(os.path.join(run_dir, ".hydra/config.yaml"),
                 cfg, optim_res=False)

    run_cfg.test = cfg.test
    run_cfg.env.show_gui = cfg.env.show_gui
    run_cfg.scene = cfg.scene
    run_cfg.target_search = cfg.target_search
    run_cfg.camera_conf = cfg.camera_conf
    max_ts = cfg.agent.learn_config.max_episode_length
    save_images = cfg.save_images

    # Load env
    env = PlayTableRL(viz=cfg.viz_obs,
                      save_images=save_images,
                      **run_cfg.env)
    env = AffordanceWrapperSim(env, max_ts,
                               affordance_cfg=run_cfg.affordance,
                               **run_cfg.env_wrapper)
    sac_cfg = {"env": env,
               "model_name": run_cfg.model_name,
               "save_dir": run_cfg.agent.save_dir,
               "net_cfg": net_cfg,
               **agent_cfg}

    run_cfg.target_search.mode = 'env'
    model = VAPOAgent(run_cfg,
                      sac_cfg=sac_cfg)
    path = "%s/trained_models/%s.pth" % (run_dir,
                                         cfg.test.model_name)
    success = model.load(path)
    if(success):
        model.evaluate(env, **cfg.test.eval_cfg)
    env.close()


if __name__ == "__main__":
    main()
