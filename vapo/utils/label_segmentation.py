import cv2
import numpy as np
from vapo.utils.img_utils import overlay_mask
import pybullet as p


# Masks generation #
def get_elipse_angle(cam_name):
    if(cam_name == "sideview_left"):
        return - 40
    elif(cam_name == "sideview_right"):
        return 45
    else:
        return 0


def delete_oclussion(mask, robot_pose):
    robot_mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    robot_mask = cv2.circle(robot_mask, robot_pose, 10, [255, 255, 255], -1)
    # robot_mask = smoothen(robot_mask, k=7)
    mask = cv2.subtract(mask, robot_mask, mask)
    # cv2.imshow("robot_mask", robot_mask)
    return mask


def create_target_mask(img, xy_coords, task, elipse_angle=0):
    mask = np.zeros((img.shape[0], img.shape[1], 1))
    color = [255, 255, 255]
    if(task == "drawer"):
        axesLength = (25, 8)  # major, minor axis
        mask = cv2.ellipse(
            mask,
            xy_coords,
            axesLength,
            elipse_angle, 0, 360, color, -1)
    else:  # Vertical handles
        axesLength = (8, 25)  # major, minor axis
        mask = cv2.ellipse(mask, xy_coords, axesLength, 0, 0, 360, color, -1)
    # mask = smoothen(mask, k=15)
    return mask


def get_img_mask_rl_agent(env, viz=False):
    cam = env.cameras[0]  # assume camera 0 is static
    rgb, _ = cam.render()

    # append 1 to get homogeneous coord "true" label
    point, _ = env.get_target_pos()  # worldpos, state
    robot_pose = env.get_state_obs()["robot_obs"][:3]
    point.append(1)
    robot_pose = np.append(robot_pose, 1)

    # Simulate gaussian noise in 3D space
    std = 0.005
    point[0] += np.random.normal(0, std)
    point[1] += np.random.normal(0, std)

    # Project point to camera
    # x, y <- pixel coords
    x, y = cam.project(point)

    # Euclidian distance on image space
    robot_x, robot_y = cam.project(robot_pose)
    # euclidian_dist = np.linalg.norm(
    #   np.array([x,y])-np.array([robot_x,robot_y]))
    # if( euclidian_dist < 20.0 ):
    #     print("Oclussion...")

    img = np.array(rgb[:, :, ::-1])
    elipse_angle = get_elipse_angle(cam.name)
    mask = create_target_mask(img, (x, y), env.task, elipse_angle)
    mask = delete_oclussion(mask, (robot_x, robot_y))
    if(viz):
        res = overlay_mask(mask, img, (0, 0, 255))
        cv2.imshow("mask", np.expand_dims(mask, -1))
        cv2.imshow("paste", res)
        cv2.waitKey(1)
    return img, mask


# From playdata
def create_circle_mask(img, xy_coords, r=10):
    mask = np.zeros((img.shape[0], img.shape[1], 1))
    color = [255, 255, 255]
    mask = cv2.circle(mask, xy_coords, radius=r, color=color, thickness=-1)
    # mask = smoothen(mask, k=15)
    return mask


def get_px_after_crop_resize(cam, px):
    tcp_x, tcp_y = px
    # Img coords after crop
    tcp_x = tcp_x - cam.crop_coords[2]
    tcp_y = tcp_y - cam.crop_coords[0]
    # Get img coords after resize
    old_w = cam.crop_coords[3] - cam.crop_coords[2]
    old_h = cam.crop_coords[1] - cam.crop_coords[0]
    tcp_x = int((tcp_x/old_w)*cam.resize_resolution[0])
    tcp_y = int((tcp_y/old_h)*cam.resize_resolution[1])
    return tcp_x, tcp_y


def get_static_mask(static_cam, static_im, point, r=10, teleop_data=False):
    # Img history containes previus frames where gripper action was open
    # Point is the point in which the gripper closed for the 1st time
    # TCP in homogeneus coord.
    point = np.append(point, 1)

    # Project point to camera
    # x,y <- pixel coords
    tcp_x, tcp_y = static_cam.project(point)
    if(teleop_data):
        tcp_x, tcp_y = get_px_after_crop_resize(static_cam, (tcp_x, tcp_y))
    static_mask = create_circle_mask(static_im, (tcp_x, tcp_y), r=r)
    return static_mask, (tcp_y, tcp_x)  # matrix coord


def get_gripper_mask(img, robot_obs, point,
                     cam=None, radius=25, teleop_data=False):
    pt, orn = robot_obs[:3], robot_obs[3:]
    if(teleop_data):
        transform_matrix = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
        transform_matrix = np.vstack([transform_matrix, np.zeros(3)])
        transform_matrix = np.hstack([transform_matrix,
                                      np.expand_dims(np.array([*pt, 1]), 0).T])
        transform_matrix = np.linalg.inv(transform_matrix)
        point = transform_matrix @ np.array([*point, 1])
        point = point[:3]

        # Transform pt to homogeneus cords and project
        tcp_x, tcp_y = cam.project(point)

        # Get img coords after resize
        tcp_x, tcp_y = get_px_after_crop_resize(cam, (tcp_x, tcp_y))
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
