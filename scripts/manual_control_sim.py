import hydra
import logging

from vapo.wrappers.play_table_rl import PlayTableRL
from vapo.wrappers.affordance.aff_wrapper_sim import AffordanceWrapperSim
from vapo.wrappers.utils import get_name
from vapo.agent.vapo_agent import VAPOAgent
from pynput import keyboard
import numpy as np
import pybullet as p

curr_key = None


def on_press(key):
    global curr_key
    if(isinstance(key, keyboard._xorg.KeyCode)):
        curr_key = str(key).replace("'", "")
    else:
        curr_key = key.name
    print("key pressed: %s" % curr_key)


def on_release(key):
    global curr_key
    curr_key = None


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()  # start to listen on a separate thread


@hydra.main(config_path="../config", config_name="cfg_playtable")
def main(cfg):
    # Auto generate names given dense, aff-mask, aff-target
    log = logging.getLogger(__name__)
    cfg.model_name = get_name(cfg, cfg.model_name)
    max_ts = cfg.agent.learn_config.max_episode_length
    cfg.viz_obs = True

    env = PlayTableRL(viz=cfg.viz_obs, **cfg.env)
    env = AffordanceWrapperSim(env, max_ts,
                               train=True,
                               affordance_cfg=cfg.affordance,
                               viz=cfg.viz_obs,
                               save_images=cfg.save_images,
                               **cfg.env_wrapper)

    sac_cfg = {"env": env,
                "eval_env": None,
                "model_name": cfg.model_name,
                "save_dir": cfg.agent.save_dir,
                "net_cfg": cfg.agent.net_cfg,
                "train_mean_n_ep": cfg.agent.train_mean_n_ep,
                "save_replay_buffer": cfg.agent.save_replay_buffer,
                "log": log,
                **cfg.agent.hyperparameters}

    log.info("model: %s" % cfg.model_name)
    model = VAPOAgent(cfg, sac_cfg=sac_cfg, wandb_login=cfg.wandb_login)

    # Manual control
    key_map = {"up": {"axis": 1, "value": 1},
                "down": {"axis": 1, "value": -1},
                "left": {"axis": 0, "value": -1},
                "right": {"axis": 0, "value": 1},
                "w": {"axis": 2, "value": 1},
                "s": {"axis": 2, "value": -1},
                "q": {"axis": 3, "value": 1},
                "e": {"axis": 3, "value": -1},
                "f": {"axis": 4, "value": -1}}
    env, s, _ = model.detect_and_correct(env, None, noisy=True)
    for i in range(10):
        for i in range(1000):
            action = np.zeros(env.action_space.shape[0])
            action[-1] = 1
            for k, v in key_map.items():
                if curr_key is not None:
                    if curr_key == k:
                        axis = v["axis"]
                        action[axis] = v["value"]
            ns, r, d, info = env.step(action)
        env, s, _ = model.detect_and_correct(env, None, noisy=True)


if __name__ == "__main__":
    main()
