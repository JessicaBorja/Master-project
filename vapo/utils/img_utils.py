import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
import vapo.utils.flowlib as flowlib
from omegaconf.listconfig import ListConfig
from vapo.utils.utils import get_transforms, torch_to_numpy


def viz_aff_centers_preds(img_obs, mask, directions,
                          object_centers, cam_type="",
                          obs_it=0, episode=None,
                          save_images=False,
                          resize=None):
    ''' C = n_classes
        img_obs: numpy array, uint8 RGB
            - shape = (H, W, 3)
            - range = (0, 255)
        mask: torch tensor, uint8
            - shape = [1, H, W]
            - range = (0, n_classes - 1)
        directions: torch tensor, float32
            - shape = [1, 2, H, W]
            - range = pixel space vectors
        object_centers: list of torch tensors, int64
            - pixel coordinates
    '''
    # To numpy
    if isinstance(mask, torch.Tensor):
        mask = torch_to_numpy(mask[0]).astype('uint8')
    if isinstance(directions, torch.Tensor):
        directions = torch_to_numpy(directions[0].permute(1, 2, 0))  # H x W x 2
    if len(object_centers) > 0 and isinstance(object_centers[0], torch.Tensor):
        object_centers = [torch_to_numpy(o) for o in object_centers]

    mask, aff_over_img, flow_over_img, _ = \
        get_aff_imgs(img_obs[:, :, ::-1], mask,
                     directions,
                     object_centers,
                     out_shape=resize)

    if(resize):
        orig_img = cv2.resize(img_obs, resize)
    cv2.imshow("flow_over_img-%s" % cam_type, flow_over_img)
    cv2.waitKey(1)

    if(save_images):
        save_dict = {}
        dct = {"%s_orig/img_%04d.png" % (cam_type, obs_it): orig_img,
               "%s_masks/img_%04d.png" % (cam_type, obs_it): mask,
               "%s_aff/img_%04d.png" % (cam_type, obs_it): aff_over_img,
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


# Treshhold between zero and one
def tresh_np(img, threshold=100):
    new_img = np.zeros_like(img)
    idx = img > threshold
    new_img[idx] = 1
    return new_img


def transform_and_predict(model, img_transforms, orig_img,
                          resize=None, class_label=None):
    '''
        Apply image transforms to input and output affordance mask and center predictions.
        :param model(torch.module): affordance model that takes the input
        :param img_transforms (ListConfig or torchvision.transforms.transforms.Compose): transforms to be applied to orig_img
        :param orig_img (numpy.ndarray): input image shape=(H,W,C) dtype='uint8'

        :return centers (list(numpy.ndarray)): list of object centers in pixel coords of the output affordance mask
        :return mask(numpy.ndarray, int64):
            - shape: [H x W]
            - range: 0 to n_classes -1
        :return directions(numpy.ndarray):
            - shape [H x W x 2]
        :return probs(numpy.ndarray): softmax output. Each channel represents a class. Channel 0 is background.
            - shape [H x W x N_Classes]
        :return initial_masks(numpy.ndarray):
            - shape:[H x W]
            - range: int values indicating object mask (0 to n_objects)
    '''
    # if(rgb):
    #     orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    # Apply validation transforms
    if isinstance(img_transforms, ListConfig):
        img_transforms = get_transforms(img_transforms, resize)

    x = torch.from_numpy(orig_img).permute(2, 0, 1).unsqueeze(0)
    x = img_transforms(x).cuda()

    # Predict affordance, centers and directions
    _, probs, aff_mask, directions = model.forward(x)

    # Filter by class
    if(class_label is not None):
        class_mask = torch.zeros_like(aff_mask)
        class_mask[aff_mask == class_label] = 1
    else:
        class_mask = aff_mask

    object_centers, directions, initial_masks = \
        model.get_centers(class_mask, directions)

    # To numpy arrays
    centers = [torch_to_numpy(o) for o in object_centers]
    probs = torch_to_numpy(probs[0].permute(1, 2, 0))  # H x W x C
    mask = torch_to_numpy(class_mask[0]).astype('uint8')  # H x W
    directions = torch_to_numpy(directions[0].permute(1, 2, 0))  # H x W x 2
    initial_masks = torch_to_numpy(initial_masks[0])  # H x W x 1
    return centers, mask, directions, probs, initial_masks


def get_aff_imgs(orig_img, mask, directions, centers,
                 out_shape=None, cam="", n_classes=2):
    '''
        :param orig_img(numpy.ndarray, uint8): BGR
            - shape:[H x W x 1]
            - range: 0-255
        :param mask(numpy.ndarray, uint8):
            - shape: [H x W]
            - range: 0 to n_classes -1
        :param directions(numpy.ndarray, float32):
            - shape [H x W x 2]
        :param centers (list(numpy.ndarray)): list of object centers in pixel coords of the output affordance mask
    '''
    pred_shape = np.array(mask.shape)
    if(out_shape is None):
        out_shape = orig_img.shape[:2]
    out_shape = np.array(out_shape)

    orig_img = cv2.resize(orig_img, out_shape)
    cm = plt.get_cmap('tab10')

    # Affordance segmentation
    if(n_classes > 2):
        aff_img = orig_img
        # Not showing background
        colors = cm(np.linspace(0, 1, n_classes-1))[:, :3]
        colors = (colors[:, ::-1] * 255).astype('uint8')
        for i in range(1, n_classes):
            obj_mask = np.zeros_like(mask)  # (img_size, img_size)
            obj_mask[mask == i] = 255
            resize_mask = cv2.resize(obj_mask, out_shape)
            aff_img = overlay_mask(resize_mask,
                                   aff_img,
                                   tuple(colors[i-1]))
        mask[mask > 0] = 255
    else:
        mask = (mask * 255).astype('uint8')
        mask = cv2.resize(mask, out_shape)
        aff_img = overlay_mask(mask, orig_img, (255, 0, 0))

    # To flow img
    flow_img = flowlib.flow_to_image(directions)  # RGB
    flow_img = flow_img[:, :, ::-1]  # BGR

    # Resize to out_shape
    flow_img = cv2.resize(flow_img, out_shape)
    mask = cv2.resize(mask, out_shape)

    # Overlay directions and centers
    flow_over_img = overlay_flow(flow_img, orig_img, mask)
    for c in centers:
        c = resize_center(c, pred_shape, out_shape)
        u, v = c[1], c[0]  # center stored in matrix convention
        flow_over_img = cv2.drawMarker(flow_over_img, (u, v),
                                       (0, 0, 0),
                                       markerType=cv2.MARKER_CROSS,
                                       markerSize=15,
                                       thickness=3,
                                       line_type=cv2.LINE_AA)
    return mask, aff_img, flow_over_img, flow_img


def resize_center(center, old_shape, new_shape):
    assert len(old_shape) == len(new_shape)
    c = np.array(center) * new_shape // old_shape
    return c


def get_px_after_crop_resize(px, crop_coords, resize_resolution):
    tcp_x, tcp_y = px
    # Img coords after crop
    tcp_x = tcp_x - crop_coords[2]
    tcp_y = tcp_y - crop_coords[0]
    # Get img coords after resize
    old_w = crop_coords[3] - crop_coords[2]
    old_h = crop_coords[1] - crop_coords[0]
    tcp_x = int((tcp_x/old_w)*resize_resolution[0])
    tcp_y = int((tcp_y/old_h)*resize_resolution[1])
    return tcp_x, tcp_y
