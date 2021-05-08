from affordance_model.utils.losses import tresh_tensor
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F


def visualize_np(mask, img, imshow=False, k=15):
    """
    Args:
        mask: np array, float64
              shape = (img_size, img_size, classes),
              between 0-255
        img: np array, uint8
             shape = (W, H, C),
             between 0-255
    return:
        res: Overlay of mask over image, np array uint8
             shape = (W, H, 3),
             between 0-255
    """
    mask = cv2.resize(mask, dsize=img.shape[:2])
    # mask = smoothen(mask, k=k)  # [0, 255] int

    res = overlay_mask(mask, img, (0, 0, 255))
    if imshow:
        # cv2.imshow("mask", np.expand_dims(mask, -1))
        # cv2.imshow("img", img)
        cv2.imshow("paste", res)
        cv2.waitKey(1)
    return res


def visualize(mask, img, imshow):
    """
    img_size is the size at which the network was trained,
    img can be of a higher size (e.g. img_size = 64, img.shape[1] = 200)
    Args:
        mask: torch tensor, shape = (1, classes, img_size, img_size), between 0-1
              after act_fnc
        img: numpy array, shape = (W, H, C), between 0-255
    return:
        res: Overlay of mask over image, shape = (W, H, 3), 0-255
    """
    if(mask.shape[1] == 1):
        mask = mask[:, 0].permute(1, 2, 0).detach().cpu().numpy()
    else:
        mask = torch.argmax(mask, axis=1).permute(1, 2, 0)
        mask = mask.detach().cpu().numpy()*255.0
    res = visualize_np(mask, img, imshow)
    return res


def overlay_flow(flow, img, mask):
    result = Image.fromarray(np.uint8(img.squeeze()))
    pil_mask = Image.fromarray(np.uint8(mask.squeeze()))
    flow = Image.fromarray(np.uint8(flow))
    result.paste(flow, (0, 0), pil_mask)
    result = np.array(result)
    return result


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


# Treshhold between zero and one
def tresh_np(img, threshold=100):
    new_img = np.zeros_like(img)
    idx = img > threshold
    new_img[idx] = 1
    return new_img
