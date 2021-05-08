# from torch.utils.data import DataLoader
import hydra
import os
import cv2
import torch
import os
import sys
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import tqdm
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
from utils.img_utils import visualize
from utils.file_manipulation import get_files
from affordance_model.segmentator import Segmentator
from affordance_model.datasets import get_transforms


@hydra.main(config_path="../config", config_name="viz_affordances")
def viz(cfg):
    # Create output directory if save_images
    if(not os.path.exists(cfg.output_dir) and cfg.save_images):
        os.makedirs(cfg.output_dir)
    # Initialize model
    run_cfg = OmegaConf.load(cfg.folder_name + "/.hydra/config.yaml")
    model_cfg = run_cfg.model_cfg
    # model_cfg = cfg.model_cfg

    checkpoint_path = os.path.join(cfg.folder_name, "trained_models")
    checkpoint_path = os.path.join(checkpoint_path, cfg.model_name)
    model = Segmentator.load_from_checkpoint(checkpoint_path, cfg=model_cfg)
    model.eval()
    print("model loaded")

    # Image needs to be multiple of 32 because of skip connections
    # and decoder layers
    img_transform = get_transforms(cfg.transforms.validation)

    # Iterate images
    files = []
    if(isinstance(cfg.data_dir, ListConfig)):
        for dir_i in cfg.data_dir:
            path = os.path.abspath(dir_i)
            if(not os.path.exists(path)):
                print("Path does not exist: %s" % path)
                continue
            files += get_files(dir_i, "jpg")
            files += get_files(dir_i, "png")
    else:
        path = os.path.abspath(cfg.data_dir)
        if(not os.path.exists(path)):
            print("Path does not exist: %s" % path)
            return
        files += get_files(cfg.data_dir, "jpg")
        files += get_files(cfg.data_dir, "png")

    for filename in tqdm.tqdm(files):
        orig_img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        # Process image as in validation
        # i.e. resize to multiple of 32, normalize
        x = torch.from_numpy(orig_img).permute(2, 0, 1).unsqueeze(0)
        x = img_transform(x)
        mask = model.predict(x)
        res = visualize(mask, orig_img, cfg.imshow)
        if(cfg.save_images):
            _, tail = os.path.split(filename)
            output_file = os.path.join(cfg.output_dir, tail)
            cv2.imwrite(output_file, res)


if __name__ == "__main__":
    viz()
