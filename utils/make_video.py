import os
import sys
import cv2
import tqdm
import json
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
from utils.file_manipulation import get_files


def make_video(files, fps=30, video_name="v"):
    h, w, c = cv2.imread(files[0]).shape
    video = cv2.VideoWriter(
                video_name,
                0x7634706d,
                fps,
                (w, h))  # 30 fps
    print("writing video to %s" % video_name)
    for f in tqdm.tqdm(files):
        img = cv2.imread(f)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()


def join_val_ep(dataset_dir, cam):
    json_file = os.path.join(dataset_dir, "episodes_split.json")
    with open(json_file) as f:
        ep_data = json.load(f)
    files = []
    for ep in ep_data["validation"].keys():
        img_folder = os.path.join(dataset_dir, ep+"/viz_out/%s_cam/" % cam)
        files += get_files(img_folder, "jpg")
    files.sort()
    return files


def make_videos(path, cam, val_dir=False):
    if(val_dir):
        video_name = os.path.join(
                    os.path.dirname(path),
                    os.path.basename(path) + "_validation.mp4")
        files = join_val_ep(path, cam)
        make_video(files, fps=30, video_name=video_name)
    else:
        for img_folder in path:
            files = get_files(img_folder, "jpg")
            if(not files):
                return
            video_name = os.path.join(
                    os.path.dirname(img_folder),
                    os.path.basename(img_folder) + ".mp4")
            make_video(files, fps=30, video_name=video_name)


if __name__ == "__main__":
    # path = ["C:/Users/Jessica/Documents/Proyecto_ssd/SAC/affordance_model/predictions/gripper_16ep_large_handles"]
    # path.append("C:/Users/Jessica/Documents/Proyecto_ssd/SAC/affordance_model/predictions/gripper_8ep_large_handles")
    # path.append("C:/Users/Jessica/Documents/Proyecto_ssd/SAC/affordance_model/predictions/gripper_4ep_large_handles")
    # path = ["C:/Users/Jessica/Documents/Proyecto_ssd/d_simg_gimg_pos_sparse"]
    path = "C:/Users/Jessica/Documents/Proyecto_ssd/datasets/vrenv_play_large_handles"
    val_dir = True
    make_videos(path, "gripper", val_dir)
