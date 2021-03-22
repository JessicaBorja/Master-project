# from torch.utils.data import DataLoader
import hydra
from affordance_model.segmentator import Segmentator
import os
import cv2
import torch
from affordance_model.utils.utils import smoothen, overlay_mask, get_transforms
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import tqdm
from utils.file_manipulation import get_files


def visualize(mask, img, imshow):
    mask = torch.argmax(mask, axis=1).permute(1, 2, 0)
    # mask = F.softmax(mask, dim = 1).squeeze(0)[1]
    mask = mask.detach().cpu().numpy()*255.0
    mask = cv2.resize(mask, dsize=img.shape[:2])
    mask = smoothen(mask, k=15)  # [0, 255] int

    res = overlay_mask(mask, img, (0, 0, 255))
    if imshow:
        # cv2.imshow("mask", np.expand_dims(mask, -1))
        cv2.imshow("img", img)
        cv2.imshow("paste", res)
        cv2.waitKey(1)
    return res

@hydra.main(config_path="./config", config_name="viz_affordances")
def viz(cfg):
    # Create output directory if save_images
    if(not os.path.exists(cfg.output_dir) and cfg.save_images):
        os.makedirs(cfg.output_dir)
    # Initialize model
    run_cfg = OmegaConf.load(cfg.folder_name + "/.hydra/config.yaml")
    model_cfg = run_cfg.model_cfg

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
            if(not os.path.exists(dir_i)):
                print("Path does not exist: %s" % dir_i)
                continue
            files += get_files(dir_i, "jpg")
            files += get_files(dir_i, "png")
    else:
        if(not os.path.exists(cfg.data_dir)):
            print("Path does not exist: %s" % cfg.data_dir)
            return
        files += get_files(cfg.data_dir, "jpg")
        files += get_files(cfg.data_dir, "png")

    for idx, filename in tqdm.tqdm(enumerate(files)):
        orig_img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        # Process image as in validation
        # i.e. resize to multiple of 32, normalize
        x = torch.from_numpy(orig_img).permute(2, 0, 1).unsqueeze(0)
        x = img_transform(x)
        mask = model(x)
        res = visualize(mask, orig_img, cfg.imshow)
        if(cfg.save_images):
            _, tail = os.path.split(filename)
            output_file = os.path.join(cfg.output_dir, tail)
            cv2.imwrite(output_file, res)


if __name__ == "__main__":
    viz()
