import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

import vapo.utils.flowlib as flowlib


def torch_to_numpy(x):
    return x.detach().cpu().numpy()


def viz_aff_centers_preds(img_obs, mask, aff_probs, directions,
                          object_centers, object_masks,
                          cam_type="", obs_it=0, episode=None,
                          save_images=False,
                          resize=None):
    ''' C = n_classes
        img_obs: numpy array, int64
            - shape = (H, W, 3)
            - range = (0, 255)
        mask: torch tensor, int64
            - shape = [1, H, W]
            - range = (0, n_classes - 1)
        aff_probs: torch tensor, float32
            - shape = [1, C, H, W]
            - range = (0, 1.0)
        directions: torch tensor, float32
            - shape = [1, 2, H, W]
            - range = pixel space vectors
        object_centers: list of torch tensors, int64
            - pixel coordinates
        object_masks: torch tensor, int64
            - shape = [1, H, W]
            - range = (0, n_objects in current image)
    '''
    # To numpy
    mask = torch_to_numpy(mask[0])  # H, W
    aff_probs = torch_to_numpy(aff_probs[0].permute(1, 2, 0))  # H, W, 2
    directions = torch_to_numpy(directions[0].permute(1, 2, 0))  # H, W, 2
    object_centers = [torch_to_numpy(o) for o in object_centers]
    object_masks = torch_to_numpy(object_masks[0])  # H, W

    # Output img
    # To flow img
    flow_img = flowlib.flow_to_image(directions)  # RGB
    flow_img = flow_img[:, :, ::-1]  # BGR

    mask_shape = mask.shape[:2]
    im_shape = img_obs.shape[:2]
    orig_img = img_obs[:, :, ::-1]

    # Multiclass
    n_classes = aff_probs.shape[-1]
    affordances = orig_img
    if(n_classes > 2):
        cm = plt.get_cmap('tab10')
        # Not showing background
        colors = cm(np.linspace(0, 1, n_classes-1))[:, :3]
        colors = (colors[:, ::-1] * 255).astype('uint8')
        for i in range(1, n_classes):
            obj_mask = np.zeros_like(mask)  # (1, img_size, img_size)
            obj_mask[mask == i] = 255
            obj_mask = cv2.resize(obj_mask.astype('float32'), im_shape)
            affordances = overlay_mask(obj_mask,
                                       affordances,
                                       tuple(colors[i-1]))
        mask[mask > 0] = 255
        mask = cv2.resize(mask.astype('float32'), im_shape)
    else:
        mask = (mask*255).astype('uint8')
        mask = cv2.resize(mask, im_shape)
        affordances = overlay_mask(mask, affordances, (255, 0, 0))

    # Reshape mask and directions
    out_img = affordances
    flow_img = cv2.resize(flow_img, im_shape)
    flow_over_img = overlay_flow(flow_img, orig_img, mask)

    # Visualize segmentations individually and obj centers
    obj_class = np.unique(object_masks)
    obj_class = obj_class[obj_class != 0]  # remove background class
    max_robustness = 0
    target_px = None
    for i, o in enumerate(object_centers):
        # Mean prob of being class 1 (foreground)
        robustness = np.mean(aff_probs[object_masks == obj_class[i], 1])
        if(robustness > max_robustness):
            max_robustness = robustness
            target_px = o

        reshaped_center = o * im_shape // mask_shape
        v, u = reshaped_center
        flow_over_img = cv2.drawMarker(flow_over_img, (u, v),
                                       (255, 255, 255),
                                       markerType=cv2.MARKER_CROSS,
                                       markerSize=15,
                                       thickness=2,
                                       line_type=cv2.LINE_AA)

    if(target_px is not None):
        reshaped_center = target_px * im_shape // mask_shape
        v, u = reshaped_center
        flow_over_img = cv2.drawMarker(flow_over_img, (u, v),
                                       (0, 0, 0),
                                       markerType=cv2.MARKER_CROSS,
                                       markerSize=15,
                                       thickness=3,
                                       line_type=cv2.LINE_AA)

    # cv2.imshow("original_img", orig_img)
    # cv2.imshow("aff_mask", mask)
    # cv2.imshow("flow_img", flow_img)
    if(resize):
        orig_img = cv2.resize(orig_img, resize)
        mask = cv2.resize(mask, resize)
        out_img = cv2.resize(out_img, resize)
        flow_over_img = cv2.resize(flow_over_img, resize)
    cv2.imshow("flow_over_img-%s" % cam_type, flow_over_img)
    cv2.waitKey(1)

    if(save_images):
        save_dict = {}
        dct = {"%s_orig/img_%04d.png" % (cam_type, obs_it): orig_img,
               "%s_masks/img_%04d.png" % (cam_type, obs_it): mask,
               "%s_aff/img_%04d.png" % (cam_type, obs_it): out_img,
               "%s_dirs/img_%04d.png" % (cam_type, obs_it): flow_over_img}
        if(episode is not None):
            for k, v in dct.items():
                save_dict["./images/ep_%04d/%s" % k] = v
        else:
            for k, v in dct.items():
                save_dict["./images/%s" % k] = v
        return save_dict
    else:
        return {}


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
        mask: torch tensor, shape = (classes, img_size, img_size), between 0-1
              after act_fnc
        img: numpy array, shape = (W, H, C), between 0-255
    return:
        res: Overlay of mask over image, shape = (W, H, 3), 0-255
    """
    if(len(mask.shape) > 3 and mask.shape[1] > 1
       or len(mask.shape) == 3 and mask.shape[0] > 1):
        mask = torch.argmax(mask, axis=1)

    mask = mask.permute(1, 2, 0).detach().cpu().numpy()
    res = visualize_np(mask*255.0, img, imshow)
    return res


def overlay_flow(flow, img, mask):
    """
    Args:
        flow: numpy array, shape = (W, H, 3), between 0 - 255
        img: numpy array, shape = (W, H, 3), between 0 - 255
        mask: numpy array, shape = (W, H), between 0 - 255
    return:
        res: Overlay of mask over image, shape = (W, H, 3), 0-255
    """
    result = Image.fromarray(np.uint8(img.squeeze()))
    pil_mask = Image.fromarray(np.uint8(mask.squeeze()))
    flow = Image.fromarray(np.uint8(flow))
    result.paste(flow, (0, 0), pil_mask)
    result = np.array(result)
    return result


def paste_img(im1, im2, mask):
    result = Image.fromarray(np.uint8(im1))
    im2 = Image.fromarray(np.uint8(im2))
    pil_mask = Image.fromarray(np.uint8(mask))
    result.paste(im2, (0, 0), pil_mask)
    result = np.array(result)
    return result


def overlay_mask(mask, img, color):
    '''
        mask: np.array
            - shape: (H, W)
            - range: 0 - 255.0
            - float32
        img: np.array
            - shape: (H, W, 3)
            - range: 0 - 255
            - uint8
        color: tuple
            - tuple size 3 RGB
            - range: 0 - 255
    '''
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
