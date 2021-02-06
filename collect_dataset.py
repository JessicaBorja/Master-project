import gym
import hydra
from omegaconf import OmegaConf
import os,sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.src.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)
from utils.env_processing_wrapper import EnvWrapper
from sac_agent.sac import SAC
from sac_agent.sac_utils.utils import EpisodeStats, tt
import cv2
import numpy as np
from PIL import Image

def save_images(imgs, directory):
    os.mkdir(directory)
    os.mkdir(directory+"/frames/")
    data_dir = directory+"/frames/"
    for ep, lst in imgs.items():
        os.mkdir("%s/%d"%(directory,ep))
        for i, im in enumerate(lst):
            filename = "%s/%d/image_%03d.jpg"%(directory,ep,i)
            cv2.imwrite(filename, im)

def transform_point(point, cam):
    #https://github.com/bulletphysics/bullet3/issues/1952
    #reshape to get homogeneus transform
    persp_m = np.array(cam.projectionMatrix).reshape((4,4)).T
    view_m = np.array(cam.viewMatrix).reshape((4,4)).T

    #Perspective proj matrix
    world_pix_tran = persp_m @ view_m @ point
    world_pix_tran =  world_pix_tran/ world_pix_tran[-1] #divide by w 
    world_pix_tran[:3] =  (world_pix_tran[:3] + 1)/2
    
    x, y = world_pix_tran[0]*cam.width, (1-world_pix_tran[1])*cam.height
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)
    return (x,y)

def gaussian(self, mean, covariance, label=None):
        """Draw 95% confidence ellipse of a 2-D Gaussian distribution.

        Parameters
        ----------
        mean : array_like
            The mean vector of the Gaussian distribution (ndim=1).
        covariance : array_like
            The 2x2 covariance matrix of the Gaussian distribution.
        label : Optional[str]
            A text label that is placed at the center of the ellipse.

        """
        vals, vecs = np.linalg.eigh(5.9915 * covariance)
        indices = vals.argsort()[::-1]
        vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

        center = int(mean[0] + .5), int(mean[1] + .5)
        axes = int(vals[0] + .5), int(vals[1] + .5)
        angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
        cv2.ellipse(
            self.image, center, axes, angle, 0, 360, self._color, 2)

def overlay_mask(mask,img):
    result = Image.fromarray(np.uint8(img))
    pil_mask = Image.fromarray(np.uint8(mask))
    color =  Image.new("RGB", result.size , (0, 255, 0))
    result.paste( color , (0, 0), pil_mask)
    result = np.array(result)
    return result

def get_seg_mask(env):
    cam = env.cameras[0]#assume camera 0 is static
    rgb, _ = cam.render()

    #append 1 to get homogeneous coord "true" label
    point, _ = env.get_target_pos()#worldpos, state
    point.append(1)

    #Transform point to pixel
    x,y = transform_point(point, cam)
    img = np.array(rgb[:, :, ::-1])
    mask = np.zeros((img.shape[0], img.shape[1], 1))
    mask = cv2.circle(mask, (x,y) , 10, [255,255,255], -1)
    mask = cv2.blur(mask, (11,11))
    
    #Overlay mask on top of image and show
    res = overlay_mask(mask,img)
    cv2.imshow("mask", mask)    
    cv2.imshow("win", img)
    cv2.imshow("paste", res)
    cv2.waitKey(1)

def init_env_and_agent(cfg):
    data_collection = cfg.data_collection
    run_cfg = OmegaConf.load(data_collection.folder_name + ".hydra/config.yaml")
    net_cfg = run_cfg.agent.net_cfg
    img_obs = run_cfg.img_obs
    env_wrapper = run_cfg.env_wrapper
    agent_cfg = run_cfg.agent.hyperparameters

    # Create evaluation environment and wrapper for the image in case there's
    # an image observation
    cfg.eval_env.task = run_cfg.task
    print(cfg.eval_env.task)
    env =  gym.make("VREnv-v0", **cfg.eval_env).env
    env =  EnvWrapper(env, **env_wrapper)

    #Load model
    path = "%s/trained_models/%s.pth"%(data_collection.folder_name, data_collection.model_name)
    print(os.path.abspath(path))
    agent = SAC(env, img_obs = img_obs, net_cfg = net_cfg, **agent_cfg)
    _ = agent.load(path)
    return env, agent

@hydra.main(config_path="./config", config_name="cfg_sac")
def collect_dataset(cfg):
    env, agent = init_env_and_agent(cfg)

    render_mode = "human" if cfg.data_collection.viz else "rgb_array"    
    imgs = {}
    segmentation = {}
    stats = EpisodeStats(episode_lengths = [], episode_rewards = [], validation_reward=[])
    for episode in range(cfg.data_collection.n_episodes):
        s = env.reset()
        episode_length, episode_reward = 0,0
        done = False
        im_lst, seg_lst = [], []
        while( episode_length < cfg.data_collection.max_timesteps and not done):
            a, _ = agent._pi.act(tt(s), deterministic = True)#sample action and scale it to action space
            a = a.cpu().detach().numpy()
            ns, r, done, info = env.step(a)
            #img = env.render("rgb_array")
            #im_lst.append(img)
            get_seg_mask(env)
            #seg_lst.append( get_seg_mask(env) )
            s = ns
            episode_reward+=r
            episode_length+=1
        imgs[episode] = im_lst
        stats.episode_rewards.append(episode_reward)
        stats.episode_lengths.append(episode_length)
    
    #save_dir = cfg.data_collection.save_dir
    #save_images(imgs, save_dir)
    env.close()

if __name__ == "__main__":
    collect_dataset()