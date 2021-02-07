import gym
from numpy import random
from omegaconf import OmegaConf
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
## File to get segmentation maks, and save data

def create_dirs(root_dir):
    frames_dir = root_dir + "/frames/"
    masks_dir = root_dir + "/masks/"
    if(not os.path.exists(root_dir)):
        os.mkdir(root_dir)
    if(not os.path.exists(frames_dir)):
        os.mkdir(frames_dir)
    if(not os.path.exists(masks_dir)):
        os.mkdir(masks_dir)
    return frames_dir, masks_dir

def save_data(data, directory):
    run_id = datetime.now().strftime('%d-%m_%H-%M')

    frames_dir, masks_dir = create_dirs(directory)
    for img, mask, name in zip(data["frames"], data["masks"], data['ids']):
        filename = "{}_{}".format(run_id, name)
        img_filename = os.path.join(frames_dir,filename) + ".jpg" 
        cv2.imwrite(img_filename, img) #Save images
        mask_filename = os.path.join(masks_dir, filename) + ".npy"
        with open(mask_filename, 'wb') as f: #Save masks
            np.save(f, mask)

def transform_point(point, cam):
    #https://github.com/bulletphysics/bullet3/issues/1952
    #reshape to get homogeneus transform
    persp_m = np.array(cam.projectionMatrix).reshape((4,4)).T
    view_m = np.array(cam.viewMatrix).reshape((4,4)).T

    #Perspective proj matrix
    world_pix_tran = persp_m @ view_m @ point
    world_pix_tran =  world_pix_tran/ world_pix_tran[-1] #divide by w 
    world_pix_tran[:3] =  (world_pix_tran[:3] + 1)/2
    
    x, y = world_pix_tran[0]*cam.width, (1-world_pix_tran[1])*cam.height
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)
    return (x,y)

def visualize_masks(mask, img):
    #Overlay mask on top of image and show
    res = overlay_mask(mask,img, color = (0,255,0))
    cv2.imshow("mask", np.expand_dims(mask,-1))    
    cv2.imshow("paste", res)
    cv2.waitKey(1)

def overlay_mask(mask, img, color):
    result = Image.fromarray(np.uint8(img))
    pil_mask = Image.fromarray(np.uint8(mask))
    color =  Image.new("RGB", result.size , color)
    result.paste( color , (0, 0), pil_mask)
    result = np.array(result)
    return result

def delete_oclussion(mask, robot_pose):
    robot_mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    robot_mask = cv2.circle(robot_mask, robot_pose, 10, [255,255,255], -1)
    robot_mask = cv2.GaussianBlur(robot_mask, (7,7), 0)
    robot_mask = cv2.normalize(robot_mask, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    mask = cv2.subtract(mask, robot_mask, mask)
    #cv2.imshow("robot_mask", robot_mask)
    return mask

def create_mask(img, xy_coords, task, elipse_angle):
    mask = np.zeros((img.shape[0], img.shape[1], 1))
    color = [255,255,255]
    if(task == "drawer"):
        axesLength = (25,8)#major, minor axis
        mask = cv2.ellipse(mask, xy_coords , axesLength, elipse_angle, 0, 360, color, -1) 
    else: #Vertical handles
        axesLength = (8,25)#major, minor axis
        mask = cv2.ellipse(mask, xy_coords , axesLength, 0, 0, 360, color, -1) 
    
    mask = cv2.GaussianBlur(mask, (15,15), 0)
    mask = cv2.normalize(mask, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return mask

def simulate_noise(x, y, std):
    x += np.random.normal(0,std)
    y += np.random.normal(0,std)
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)
    return (x,y)

def get_elipse_angle(cam_name):
    if(cam_name == "sideview_left"):
        return - 40
    elif(cam_name == "sideview_right"):
        return 45
    else:
        return 0

def get_img_mask_pair(env, viz=False):
    cam = env.cameras[0]#assume camera 0 is static
    rgb, _ = cam.render()

    #append 1 to get homogeneous coord "true" label
    point, _ = env.get_target_pos()#worldpos, state
    robot_pose = env.get_state_obs()["robot_obs"][:3]
    point.append(1)
    robot_pose = np.append(robot_pose,1)

    #Transform point to pixel
    x,y = transform_point(point, cam)
    x,y = simulate_noise(x, y, std = 0.01) #simulate zero-mean gaussian noise

    #Euclidian distance on image space
    robot_x, robot_y = transform_point( robot_pose, cam)
    # euclidian_dist = np.linalg.norm(np.array([x,y])-np.array([robot_x,robot_y])) 
    # if( euclidian_dist < 20.0 ):
    #     print("Oclussion...")

    img = np.array(rgb[:, :, ::-1])
    elipse_angle = get_elipse_angle(cam.name)
    mask = create_mask(img, (x,y), env.task, elipse_angle)
    mask = delete_oclussion(mask, (robot_x,robot_y))
    if(viz):
        visualize_masks(mask,img)
    return img, mask