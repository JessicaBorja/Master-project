import cv2
import numpy as np
from affordance_model.utils.utils import overlay_mask, smoothen
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
    robot_mask = smoothen(robot_mask, k=7)
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
    # mask dtype = float [0., 255.]
    mask = smoothen(mask, k=15)
    return mask


def transform_point(point, cam):
    # https://github.com/bulletphysics/bullet3/issues/1952
    # reshape to get homogeneus transform
    persp_m = np.array(cam.projectionMatrix).reshape((4, 4)).T
    view_m = np.array(cam.viewMatrix).reshape((4, 4)).T

    # Perspective proj matrix
    world_pix_tran = persp_m @ view_m @ point
    world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
    world_pix_tran[:3] = (world_pix_tran[:3] + 1)/2
    x, y = world_pix_tran[0]*cam.width, (1-world_pix_tran[1])*cam.height
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)
    return (x, y)


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
    # x,y <- pixel coords
    x, y = transform_point(point, cam)

    # Euclidian distance on image space
    robot_x, robot_y = transform_point(robot_pose, cam)
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
    mask = smoothen(mask, k=15)
    return mask


def get_static_mask(static_cam, static_im, point):
    # Img history containes previus frames where gripper action was open
    # Point is the point in which the gripper closed for the 1st time
    # TCP in homogeneus coord.
    point = np.append(point, 1)

    # Project point to camera
    # x,y <- pixel coords
    tcp_x, tcp_y = transform_point(point, static_cam)
    static_mask = create_circle_mask(static_im, (tcp_x, tcp_y), r=10)
    return static_mask


def get_gripper_mask(img, robot_obs, point, cam_properties=None, radius=25):
    pt, orn = robot_obs[:3], robot_obs[3:]
    orn = p.getQuaternionFromEuler(orn)
    cam2tcp_pos = [0.1, 0, -0.1]
    cam2tcp_orn = [0.430235, 0.4256151, 0.559869, 0.5659467]
    cam_pos, cam_orn = p.multiplyTransforms(
                                pt, orn,
                                cam2tcp_pos, cam2tcp_orn)

    # tcp_x, tcp_y = img.shape[0]//2, img.shape[1]//2
    # mask = create_circle_mask(img, (tcp_x, tcp_y), r=radius)

    # Create projection and view matrix
    cam_rot = p.getMatrixFromQuaternion(cam_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    view_matrix = p.computeViewMatrix(cam_pos, cam_pos + cam_rot_y, -cam_rot_z)
    projection_matrix = p.computeProjectionMatrixFOV(
                                **cam_properties["proj_matrix"])

    # Create camera obj since transform_point receives and obj
    gripper_cam = type('obj', (object,),
                       {'projectionMatrix': projection_matrix,
                        'viewMatrix': view_matrix,
                        "width": cam_properties["width"],
                        "height": cam_properties["height"]
                        })()

    # Transform pt to homogeneus cords and project
    point = np.append(point, 1)
    tcp_x, tcp_y = transform_point(point, gripper_cam)
    mask = create_circle_mask(img, (tcp_x, tcp_y), r=radius)
    return mask
