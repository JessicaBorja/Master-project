import numpy as np
import cv2
from PIL import Image
import torch


def visualize(mask, img, imshow):
    # mask = Torch.tensor(), shape[1, 2, H, W]
    # img = numpy, shape[1, 1, H, W]
    if(mask.shape[1] == 1):
        mask = mask[:, 0].permute(1, 2, 0).detach().cpu().numpy()
    else:
        mask = torch.argmax(mask, axis=1).permute(1, 2, 0)
        mask = mask.detach().cpu().numpy()*255.0
    mask = cv2.resize(mask, dsize=img.shape[:2])
    mask = smoothen(mask, k=15)  # [0, 255] int

    res = overlay_mask(mask, img, (0, 0, 255))
    if imshow:
        # cv2.imshow("mask", np.expand_dims(mask, -1))
        # cv2.imshow("img", img)
        cv2.imshow("paste", res)
        cv2.waitKey(1)
    return res


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
