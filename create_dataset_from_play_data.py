import hydra
import os,sys, glob
from utils.env_processing_wrapper import EnvWrapper
from utils.data_collection import *
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import tqdm
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")

@hydra.main(config_path="./config", config_name="cfg_datacollection")
def collect_dataset(cfg):
    save_static = { "frames":[], "masks":[], "ids":[]}
    save_gripper = { "frames":[], "masks":[], "ids":[]}

    # Instantiate camera to get projection and view matrices
    static_cam = hydra.utils.instantiate(cfg.env.cameras[0], cid=0, robot_id=None, objects=None)

    # Iterate rendered_data
    files = glob.glob(cfg.play_data_dir + "/*.npz")
    files.sort()
    static_hist = []
    past_action = 1
    history_length, segment_k = 50, 10
    for idx, filename in tqdm.tqdm(enumerate(files)):
        try:
            data = np.load(filename, allow_pickle=True)
            if( len(data['rgb_static'].shape)!=3 or len(data['rgb_gripper'].shape)!=3):
                 raise Exception("Corrupt data")
        except Exception as e:
            #print(e)
            data = None
            #print("cannot load file as numpy compressed: %s"%filename)
        if(data is not None):
            # Keep at most 10 elements in list
            static_hist.append( data['rgb_static'][:, :, ::-1] )
            if( len(static_hist) >= history_length ):
                static_hist.pop(0)
            if( data['actions'][-1] == 0 ):
                # Get mask for current image
                gripper = data['rgb_gripper'][:, :, ::-1] # Get last image
                gripper_mask = get_gripper_mask( gripper, radius = 25)
                if(cfg.viz):
                    gripper_out = overlay_mask(gripper_mask, gripper, (0,0,255))
                    cv2.imshow("Gripper", gripper_out)
                    cv2.waitKey(1)
                head, tail = os.path.split(filename) 
                save_gripper["frames"].append(gripper)
                save_gripper["masks"].append(gripper_mask)
                save_gripper["ids"].append( "gripper_%s"%(tail[:-4]))    

                # Get mask for static images
                if(past_action == 1):
                    # Save static cam masks
                    point = data['robot_obs'][:3]
                    static_lst = static_hist[:segment_k]
                    static_masks = get_static_mask(static_cam, static_lst, point)
                    for idx, (static, static_mask) in enumerate( zip( static_lst, static_masks)):
                        static_out = overlay_mask(static_mask, static, (0,0,255))
                        if(cfg.viz):
                            cv2.imshow("Static", static_out)
                            cv2.waitKey(1)
                        # get filename only w/o extension
                        head, tail = os.path.split(filename) 
                        save_static["frames"].append(static)
                        save_static["masks"].append(static_mask)
                        save_static["ids"].append( "static_%s_%d"%(tail[:-4], idx))
                    static_hist, static_lst = [], []
            
            if ( len(save_gripper['frames']) + len(save_static['frames']) > 150):
                save_data(save_static, cfg.save_dir, sub_dir = "static_cam")
                save_data(save_gripper, cfg.save_dir, sub_dir = "gripper_cam")
                save_static = { "frames":[], "masks":[], "ids":[]}
                save_gripper = { "frames":[], "masks":[], "ids":[]}
            past_action = data['actions'][-1]     


    save_data(save_static, cfg.save_dir, sub_dir = "static_cam")
    save_data(save_gripper, cfg.save_dir, sub_dir = "gripper_cam")
    create_data_split(cfg.save_dir)

if __name__ == "__main__":
    collect_dataset()