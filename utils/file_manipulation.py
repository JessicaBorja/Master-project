import os
import glob
import json
import tqdm
import numpy as np
import cv2


# Select files that have a segmentation mask
def select_files(data, split, ep, episode_files,
                 remove_blank_masks, skip_first_frames=False):
    # Skip first n files for validation s.t. masks are
    # better representatives of true affordances
    # For the static_cam
    if skip_first_frames and split == "validation":
        episode_files.sort()
        fraction = len(episode_files)//3  # Discard first third
        iter_files = episode_files[fraction:]
    else:
        iter_files = episode_files

    for file in tqdm.tqdm(iter_files):
        head, tail = os.path.split(file)
        # Remove extension name
        file_name = tail.split('.')[0]
        # Last folder
        file_relative_path = os.path.basename(os.path.normpath(head))
        file_name = os.path.join(file_relative_path, file_name)
        if(remove_blank_masks):
            mask = np.load(file)["mask"]  # (H, W)
            if(mask.max() > 0):  # at least one pixel is not background
                data[split]['episode_%d' % ep].append(file_name)
        else:
            data[split]['episode_%d' % ep].append(file_name)
    return data


# Split episodes into train and validation
def create_data_ep_split(root_dir, remove_blank_mask_instances=True):
    # Episodes are subdirectories
    n_episodes = len(glob.glob(root_dir + "/*/"))
    # Split data
    data = {"train": [], "validation": []}
    # val_ep = np.random.choice(n_episodes, 3, replace=False)
    val_ep = [1, 10, 13]
    train_ep = [ep for ep in range(n_episodes) if ep not in val_ep]
    data["validation"] = {"episode_%d" % e: [] for e in val_ep}
    data["train"] = {"episode_%d" % e: [] for e in train_ep}

    for ep in tqdm.tqdm(range(n_episodes)):
        ep_dir = os.path.join(root_dir, "episode_%d" % ep)
        split = "validation" if ep in val_ep else "train"

        gripper_cam_files = glob.glob("%s/data/*/*gripper*" % ep_dir)
        data = select_files(data, split, ep, gripper_cam_files,
                            remove_blank_mask_instances)

        static_cam_files = glob.glob("%s/data/*/*static*" % ep_dir)
        data = select_files(data, split, ep, static_cam_files,
                            remove_blank_mask_instances,
                            skip_first_frames=True)

    with open(root_dir+'/episodes_split.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)


# Create directories if not exist
def create_dirs(root_dir, sub_dir, directory_lst):
    dir_lst = [root_dir]
    for d_name in directory_lst:
        dir_lst.append(root_dir + "/%s/%s/" % (d_name, sub_dir))

    for directory in dir_lst:
        if(not os.path.exists(directory)):
            os.makedirs(directory)
    dir_lst.pop(0)  # Remove root_dir
    return dir_lst


# Save a directory wtih frames, masks, viz_out
def save_data(data_dict, directory, sub_dir, save_viz=True):
    if(save_viz):
        data_dir, viz_out_dir = \
            create_dirs(directory, sub_dir, ['data', 'viz_out'])
    else:
        data_dir = create_dirs(directory, sub_dir, ['data'])
    for img_id, img_dict in data_dict.items():
        # Write vizualization output
        if(save_viz):
            img_filename = os.path.join(viz_out_dir, img_id) + ".jpg"
            cv2.imwrite(img_filename, img_dict['viz_out'])  # Save images
            img_dict.pop('viz_out')

        # img_dict = {"frame":np.array, "mask":np.array, "centers": np.array}
        # frame is in BGR
        filename = os.path.join(data_dir, img_id) + ".npz"
        np.savez_compressed(
            filename,
            **img_dict
        )


# Ger valid numpy files with raw data
def get_files(path, extension):
    if(not os.path.isdir(path)):
        print("path does not exist: %s" % path)
    files = glob.glob(path + "/*.%s" % extension)
    if not files:
        print("No *.%s files found in %s" % (extension, path))
    files.sort()
    return files


def viz_rendered_data(path):
    # Iterate images
    files = glob.glob(path + "/*.npz")
    for idx, filename in enumerate(files):
        try:
            data = np.load(filename, allow_pickle=True)
            cv2.imshow("static", data['rgb_static'][:, :, ::-1])  # W, H, C
            cv2.imshow("gripper", data['rgb_gripper'][:, :, ::-1])  # W, H, C
            cv2.waitKey(0)
            # tcp pos(3), euler angles (3), gripper_action(0 close - 1 open)
            print(data['actions'])
        except Exception as e:
            print("[ERROR] %s: %s" % (str(e), filename))
