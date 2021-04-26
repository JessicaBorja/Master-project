import gym
import hydra
from omegaconf import OmegaConf
from env_wrappers.env_wrapper import ObservationWrapper
from utils.label_segmentation import get_img_mask_rl_agent
from utils.file_manipulation import create_data_ep_split, save_data
from sac_agent.sac import SAC
from sac_agent.sac_utils.utils import EpisodeStats, tt
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


def init_env_and_agent(cfg):
    # Load agent data from agent path
    run_cfg = OmegaConf.load(cfg.folder_name + ".hydra/config.yaml")
    net_cfg = run_cfg.agent.net_cfg
    img_obs = run_cfg.img_obs
    env_wrapper = run_cfg.env_wrapper
    agent_cfg = run_cfg.agent.hyperparameters

    # Create evaluation environment and wrapper for the image in case there's
    # an image observation
    cfg.env.task = run_cfg.task
    cfg.env.data_path = cfg.vrenv_data_path
    print(cfg.env.task)
    env = gym.make("VREnv-v0", **cfg.env).env
    env = ObservationWrapper(env, **env_wrapper)

    # Load model
    path = "%s/trained_models/%s.pth" % (cfg.folder_name, cfg.model_name)
    print(os.path.abspath(path))
    agent = SAC(env, img_obs=img_obs, net_cfg=net_cfg, **agent_cfg)
    success = agent.load(path)
    return env, agent, success


@hydra.main(config_path="./config", config_name="cfg_datacollection")
def collect_dataset(cfg):
    env, agent, success_load = init_env_and_agent(cfg)
    if(not success_load):
        print("False agent path.. ending execution")
        env.close()
        return

    data = {}
    stats = EpisodeStats(
        episode_lengths=[],
        episode_rewards=[],
        validation_reward=[])
    for episode in range(cfg.n_episodes):
        s = env.reset()
        episode_length, episode_reward = 0, 0
        done = False
        print("Running episode %d" % episode)
        while(episode_length < cfg.max_timesteps and not done):
            a, _ = agent._pi.act(tt(s), deterministic=True)
            a = a.cpu().detach().numpy()
            ns, r, done, _ = env.step(a)
            img, mask = get_img_mask_rl_agent(env, cfg.viz)
            im_id = "%s_%d_image%04d" % (env.task, episode, episode_length)
            data[im_id] = {"frame": img, "mask": mask}
            s = ns
            episode_reward += r
            episode_length += 1
        stats.episode_rewards.append(episode_reward)
        stats.episode_lengths.append(episode_length)

    save_dir = cfg.save_dir
    save_data(data, save_dir, sub_dir=env.task, save_viz=False)
    create_data_ep_split(save_dir)
    env.close()


if __name__ == "__main__":
    collect_dataset()
