import os, glob
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import json
from affordance_model.utils.utils import *

#######################################################
#### File to get segmentation maks, and save data #####
#######################################################
def create_data_split(root_dir):
    data = {'train':[],"validation":[]}
    all_files = glob.glob(root_dir+"/frames/*")
    #Get a third of the data as validation
    val_idx = np.random.randint(low=0, high= len(all_files), size= len(all_files)//3 ) 

    for idx, file in enumerate(all_files):
        _, tail = os.path.split(file)
        file_name = tail.split('.')[0]
        if(idx in val_idx): #Validation
            data['validation'].append(file_name)
        else:
            data['train'].append(file_name)

    with open(root_dir+'/data.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)

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

#######################################################
############### Masks generation ######################
#######################################################
def get_elipse_angle(cam_name):
    if(cam_name == "sideview_left"):
        return - 40
    elif(cam_name == "sideview_right"):
        return 45
    else:
        return 0

def delete_oclussion(mask, robot_pose):
    robot_mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    robot_mask = cv2.circle(robot_mask, robot_pose, 10, [255,255,255], -1)
    robot_mask = smoothen(robot_mask, k=7)
    mask = cv2.subtract(mask, robot_mask, mask)
    #cv2.imshow("robot_mask", robot_mask)
    return mask

def create_target_mask(img, xy_coords, task, elipse_angle):
    mask = np.zeros((img.shape[0], img.shape[1], 1))
    color = [255,255,255]
    if(task == "drawer"):
        axesLength = (25,8)#major, minor axis
        mask = cv2.ellipse(mask, xy_coords , axesLength, elipse_angle, 0, 360, color, -1) 
    else: #Vertical handles
        axesLength = (8,25)#major, minor axis
        mask = cv2.ellipse(mask, xy_coords , axesLength, 0, 0, 360, color, -1) 
    # mask dtype = float [0., 255.]
    mask = smoothen(mask, k=15)
    return mask

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

def get_img_mask_pair(env, viz=False):
    cam = env.cameras[0]#assume camera 0 is static
    rgb, _ = cam.render()

    #append 1 to get homogeneous coord "true" label
    point, _ = env.get_target_pos()#worldpos, state
    robot_pose = env.get_state_obs()["robot_obs"][:3]
    point.append(1)
    robot_pose = np.append(robot_pose,1)

    # Simulate gaussian noise in 3D space
    std = 0.005
    point[0] += np.random.normal(0,std)
    point[1] += np.random.normal(0,std)
    
    # Project point to camera
    # x,y <- pixel coords
    x,y = transform_point(point, cam)

    #Euclidian distance on image space
    robot_x, robot_y = transform_point( robot_pose, cam)
    # euclidian_dist = np.linalg.norm(np.array([x,y])-np.array([robot_x,robot_y])) 
    # if( euclidian_dist < 20.0 ):
    #     print("Oclussion...")

    img = np.array(rgb[:, :, ::-1])
    elipse_angle = get_elipse_angle(cam.name)
    mask = create_target_mask(img, (x,y), env.task, elipse_angle)
    mask = delete_oclussion(mask, (robot_x,robot_y))
    if(viz):
        visualize_masks(mask,img)
    return img, mask