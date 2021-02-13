import numpy as np
import cv2
from PIL import Image

def overlay_mask(mask, img, color):
    result = Image.fromarray(np.uint8(img))
    pil_mask = Image.fromarray(np.uint8(mask))
    color =  Image.new("RGB", result.size , color)
    result.paste( color , (0, 0), pil_mask)
    result = np.array(result)
    return result

def smoothen(img, k):
    # img.shape = W, H, C
    img = cv2.GaussianBlur(img, (k,k), 0)
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return img