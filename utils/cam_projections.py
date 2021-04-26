import numpy as np


def world2pixel(point, cam):
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


def pixel2world(cam, u, v, depth):
    T_world_cam = np.linalg.inv(np.array(cam.viewMatrix).reshape((4, 4)).T)

    z = depth[v, u]
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    x = (u - cam.width // 2) * z / foc
    y = -(v - cam.height // 2) * z / foc
    z = -z
    world_pos = (T_world_cam @ np.array([x, y, z, 1]))[:3]
    return world_pos
