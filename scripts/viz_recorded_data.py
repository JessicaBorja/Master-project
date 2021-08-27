import hydra
import cv2
import numpy as np
import tqdm
import os
import sys
from omegaconf import OmegaConf

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir+"/VREnv/")
from utils.file_manipulation import get_files, check_file


def viz_data(cfg):
    # Episodes info
    # Simulation
    files = get_files(cfg.play_data_dir, "npz")  # Sorted files
    if(not cfg.teleop_data):
        # ep_lens = np.load(os.path.join(cfg.play_data_dir, "ep_lens.npy"))
        ep_start_end_ids = np.load(os.path.join(
            cfg.play_data_dir,
            "ep_start_end_ids.npy"))
        end_ids = ep_start_end_ids[:, -1]
    else:
        # Real life experiments
        files.remove(os.path.join(cfg.play_data_dir, "camera_info.npz"))

    for idx, filename in enumerate(tqdm.tqdm(files)):
        data = check_file(filename)
        if(data is None):
            continue  # Skip file

        new_size = (400, 400)
        for key in ['rgb_static', 'depth_static', 'rgb_gripper', 'depth_gripper']:
            img = cv2.resize(data[key], new_size)
            if('rgb' in key):
                cv2.imshow(key, img[:, :, ::-1])
            else:
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                cv2.imshow(key, img)

        cv2.waitKey(1)


@hydra.main(config_path="../config", config_name="cfg_datacollection")
def main(cfg):
    viz_data(cfg)


if __name__ == "__main__":
    main()
