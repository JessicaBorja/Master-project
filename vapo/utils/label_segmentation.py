from hashlib import new
import cv2
import numpy as np
import pybullet as p
from vapo.utils.img_utils import resize_center, get_px_after_crop_resize


def resize_mask_and_center(mask, center, new_size):
    old_size = mask.shape[:2]
    mask = cv2.resize(mask, new_size)
    new_center = resize_center(center, old_size, new_size)
    return mask, new_center


# From playdata
def create_circle_mask(img, xy_coords, r=10):
    mask = np.zeros((img.shape[0], img.shape[1], 1))
    color = [255, 255, 255]
    mask = cv2.circle(mask, xy_coords, radius=r, color=color, thickness=-1)
    return mask


def get_static_mask(cam, static_im, point, r=10, teleop_data=False):
    # Img history containes previus frames where gripper action was open
    # Point is the point in which the gripper closed for the 1st time
    # TCP in homogeneus coord.
    point = np.append(point, 1)

    # Project point to camera
    # x,y <- pixel coords
    tcp_x, tcp_y = cam.project(point)
    if(teleop_data):
        tcp_x, tcp_y = get_px_after_crop_resize((tcp_x, tcp_y),
                                                cam.crop_coords,
                                                cam.resize_resolution)
    static_mask = create_circle_mask(static_im, (tcp_x, tcp_y), r=r)
    return static_mask, (tcp_y, tcp_x)  # matrix coord


def tcp_to_global(pos, orn, offset, homogeneous=False):
    transform_matrix = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
    transform_matrix = np.vstack([transform_matrix, np.zeros(3)])
    transform_matrix = np.hstack([transform_matrix,
                                  np.expand_dims(np.array([*pos, 1]), 0).T])
    glob_offset = transform_matrix @ np.array([*offset, 1])
    if(not homogeneous):
        glob_offset = glob_offset[:3]
    return glob_offset


def get_gripper_mask(img, robot_obs, point, offset=[0, 0, 0],
                     cam=None, radius=25, teleop_data=False):
    pt, orn = robot_obs[:3], robot_obs[3:]
    if(teleop_data):
        transform_matrix = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
        transform_matrix = np.vstack([transform_matrix, np.zeros(3)])
        tcp2global = np.hstack([transform_matrix,
                                np.expand_dims(np.array([*pt, 1]), 0).T])
        global2tcp = np.linalg.inv(tcp2global)
        point = global2tcp @ np.array([*point, 1])
        tcp_pt = point[:3] + offset

        # Transform pt to homogeneus cords and project
        tcp_x, tcp_y = cam.project(tcp_pt)

        # Get img coords after resize
        tcp_x, tcp_y = get_px_after_crop_resize((tcp_x, tcp_y),
                                                cam.crop_coords,
                                                cam.resize_resolution)

    else:
        orn = p.getQuaternionFromEuler(orn)
        tcp2cam_pos, tcp2cam_orn = cam.tcp2cam_T
        # cam2tcp_pos = [0.1, 0, -0.1]
        # cam2tcp_orn = [0.430235, 0.4256151, 0.559869, 0.5659467]
        cam_pos, cam_orn = p.multiplyTransforms(
                                    pt, orn,
                                    tcp2cam_pos, tcp2cam_orn)

        # Create projection and view matrix
        cam_rot = p.getMatrixFromQuaternion(cam_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]

        # Extrinsics change as robot moves
        cam.viewMatrix = p.computeViewMatrix(cam_pos,
                                             cam_pos + cam_rot_y,
                                             - cam_rot_z)

        # Transform pt to homogeneus cords and project
        point = np.append(point, 1)
        tcp_x, tcp_y = cam.project(point)

    if(tcp_x > 0 and tcp_y > 0):
        mask = create_circle_mask(img, (tcp_x, tcp_y), r=radius)
    else:
        mask = np.zeros((img.shape[0], img.shape[1], 1))
    return mask, (tcp_y, tcp_x)
