import hydra
import os,sys, glob

from numpy.lib.ufunclike import fix
from utils.env_processing_wrapper import EnvWrapper
from utils.data_collection import *
import cv2
import numpy as np
import tqdm
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")

# Keep points in a distance larger than radius from new_point
# Do not keep fixed points more than 100 frames
def update_fixed_points(fixed_points, new_point, current_frame_idx, radius = 0.1): 
    x = []
    for frame_idx, p in fixed_points:
        if np.linalg.norm( new_point - p ) > radius\
            and current_frame_idx - frame_idx < 1500:
            x.append( (frame_idx, p) )
    # x = [ p for (frame_idx, p) in fixed_points if ( np.linalg.norm( new_point - p ) > radius)] # and current_frame_idx - frame_idx < 100 )
    return x

def update_mask(fixed_points, mask, frame_img_tuple, cam):
    # Update masks with fixed_points
    (frame_timestep, img) = frame_img_tuple
    for point_timestep, p in fixed_points:
        if( frame_timestep >= point_timestep): # Only add point if it was fixed before seing img
            new_mask = get_static_mask(cam, img , p)
            mask = overlay_mask(new_mask, mask, (255, 255, 255))
    return mask

def check_file(filename):
    try:
        data = np.load(filename, allow_pickle=True)
        if( len(data['rgb_static'].shape)!=3 or len(data['rgb_gripper'].shape)!=3):
                raise Exception("Corrupt data")
    except Exception as e:
        #print(e)
        data = None
    return data

def collect_dataset_close(cfg):
    save_static, save_gripper = {}, {}
    # Instantiate camera to get projection and view matrices
    static_cam = hydra.utils.instantiate(cfg.env.cameras[0], cid=0, robot_id=None, objects=None)

    # Iterate rendered_data
    files = get_files( cfg.play_data_dir, "npz")
    static_hist = []
    past_action = 1
    # Will segment 40 frames
    back_frames_max = 50
    back_frames_min = 10
    for idx, filename in tqdm.tqdm(enumerate(files)):
        data = check_file(filename)
        if(data is None): 
            # Skip file
            continue

        # Keep at most 10 elements in list
        _, tail = os.path.split(filename) 
        img_id = "static_%s"%(tail[:-4])
        static_hist.append( (img_id, data['rgb_static'][:, :, ::-1]) )
        
        # Start of interaction
        if( data['actions'][-1] == 0 ):
            # Get mask for gripper image
            gripper = data['rgb_gripper'][:, :, ::-1] # Get last image
            gripper_mask = get_gripper_mask( gripper, radius = 25)
            if(cfg.viz):
                gripper_out = overlay_mask(gripper_mask, gripper, (0,0,255))
                cv2.imshow("Gripper", gripper_out)
                cv2.waitKey(1)
            _, tail = os.path.split(filename) 
            img_id = "gripper_%s"%(tail[:-4])
            save_gripper[img_id] = { "frame": gripper, "mask": gripper_mask, "viz_out": gripper_out } 

            # Get mask for static cam images
            if(past_action == 1):
                # Save static cam masks
                point = data['robot_obs'][:3]
                for idx, (static_id, static_im) in enumerate( static_hist ):
                    if(idx <= len(static_hist) - back_frames_min and idx >  len(static_hist) - back_frames_max):
                        static_mask = get_static_mask(static_cam, static_im, point)
                        static_out = overlay_mask(static_mask, static_im, (0,0,255))
                    else: 
                        # No segmentation in current image due to oclusion
                        static_mask = np.zeros( static_im.shape[:2] )
                        static_out = static_im

                    if(cfg.viz):
                        cv2.imshow("Static", static_out)
                        cv2.waitKey(1)
                    # get filename only w/o extension
                    save_static[static_id] = {"frame": static_im, "mask": static_mask, "viz_out": static_out}
                static_hist = []
        
        # Save data every 150 image-mask pair to avoid filling up RAM memory
        if ( len(save_gripper.keys()) + len(save_static.keys()) >= 100 ):
            save_data(save_static, cfg.save_dir, sub_dir = "static_cam")
            save_data(save_gripper, cfg.save_dir, sub_dir = "gripper_cam")
            save_static, save_gripper = {}, {}

        # Update past action
        past_action = data['actions'][-1]     

    save_data(save_static, cfg.save_dir, sub_dir = "static_cam")
    save_data(save_gripper, cfg.save_dir, sub_dir = "gripper_cam")
    create_data_split(cfg.save_dir)

def collect_dataset_close_open(cfg):
    save_static, save_gripper = {}, {}

    # Instantiate camera to get projection and view matrices
    static_cam = hydra.utils.instantiate(cfg.env.cameras[0], cid=0, robot_id=None, objects=None)

    # Iterate rendered_data
    files = get_files( cfg.play_data_dir, "npz")
    static_hist , fixed_points = [], []
    past_action = 1
    frame_idx = 0
    # Will segment 40 frames
    back_frames_max = 50
    back_frames_min = 10
    for idx, filename in tqdm.tqdm(enumerate(files)):
        data = check_file(filename)
        if(data is None): 
            # Skip file
            continue
        
        # Initialize img, mask, id
        _, tail = os.path.split(filename) 
        img_id = "static_%s"%(tail[:-4])
        img = data['rgb_static'][:, :, ::-1]
        static_hist.append( (frame_idx, img_id, img) )
        frame_idx += 1

        # Start of interaction
        if( data['actions'][-1] == 0 ): # closed gripper
            # Start of interaction
            # Get mask for current image
            gripper = data['rgb_gripper'][:, :, ::-1] # Get last image
            gripper_mask = get_gripper_mask( gripper, radius = 25)
            gripper_out = overlay_mask(gripper_mask, gripper, (0,0,255))
            if(cfg.viz):
                cv2.imshow("Gripper", gripper_out)
                cv2.waitKey(1)
            _, tail = os.path.split(filename) 
            img_id =  "gripper_%s"%(tail[:-4]) 
            save_gripper[img_id] = { "frame": gripper, "mask": gripper_mask, "viz_out": gripper_out } 

            # Get mask for static images
            #past_action == 1 and current == 0
            if(past_action == 1):
                # If region was already labeled, delete previous point
                point = data['robot_obs'][:3]
                fixed_points = update_fixed_points(fixed_points, point, frame_idx)
                
                # Save static cam masks
                for idx, (fr_idx, im_id, img) in enumerate( static_hist ):
                    if(idx <= len(static_hist) - back_frames_min and idx >  len(static_hist) - back_frames_max):
                        mask = get_static_mask(static_cam, img, point) # Get new grip
                    else: 
                        # No segmentation in current image due to occlusion
                        mask = np.zeros( img.shape[:2] )
                    #static_mask = update_mask(fixed_points, static_mask, (fr_idx, img), static_cam) # Add fixed points
                    fp_mask = update_mask(fixed_points, np.zeros_like(mask), (fr_idx, img), static_cam) 
                    out_separate = overlay_mask(fp_mask, img, (255,0,0))
                    out_separate = overlay_mask(mask, out_separate, (0,0,255))
                    
                    # Real mask
                    mask = overlay_mask(fp_mask, mask, (255, 255, 255))
                    out_img = overlay_mask(mask, img, (0,0,255))
                    
                    if(cfg.viz):
                        cv2.imshow("Separate", out_separate)
                        cv2.imshow("Real", out_img)
                        cv2.waitKey(1)
                    save_static[im_id] = {"frame": img, "mask": mask, "viz_out": out_separate}
                static_hist = []
        elif( past_action == 0):
            # Closed -> open transition
            curr_point = data['robot_obs'][:3]
            fixed_points.append( (frame_idx, curr_point) )

        if ( len(save_gripper.keys()) + len(save_static.keys()) > 150):
            save_data(save_static, cfg.save_dir, sub_dir = "static_cam")
            save_data(save_gripper, cfg.save_dir, sub_dir = "gripper_cam")
            save_static, save_gripper = {}, {}
        past_action = data['actions'][-1]     

    save_data(save_static, cfg.save_dir, sub_dir = "static_cam")
    save_data(save_gripper, cfg.save_dir, sub_dir = "gripper_cam")
    create_data_split(cfg.save_dir)

@hydra.main(config_path="./config", config_name="cfg_datacollection")
def main(cfg):
    #create_data_split(cfg.save_dir)
    #collect_dataset_close(cfg)
    collect_dataset_close_open(cfg)

if __name__ == "__main__":
    main()