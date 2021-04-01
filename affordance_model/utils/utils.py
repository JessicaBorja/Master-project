import numpy as np
import cv2
from PIL import Image
from affordance_model.datasets import VREnvData
from torch.utils.data import DataLoader


def get_loaders(logger, dataset_cfg, dataloader_cfg):
    train = VREnvData(split="train", log=logger, **dataset_cfg)
    val = VREnvData(split="validation", log=logger, **dataset_cfg)
    logger.info('train_data {}'.format(train.__len__()))
    logger.info('val_data {}'.format(val.__len__()))

    train_loader = DataLoader(train, shuffle=True, **dataloader_cfg)
    val_loader = DataLoader(val, **dataloader_cfg)
    logger.info('train minibatches {}'.format(len(train_loader)))
    logger.info('val minibatches {}'.format(len(val_loader)))
    return train_loader, val_loader


def overlay_mask(mask, img, color):
    result = Image.fromarray(np.uint8(img))
    pil_mask = Image.fromarray(np.uint8(mask))
    color = Image.new("RGB", result.size, color)
    result.paste(color, (0, 0), pil_mask)
    result = np.array(result)
    return result


def smoothen(img, k):
    # img.shape = W, H, C
    img = cv2.GaussianBlur(img, (k, k), 0)
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return img
