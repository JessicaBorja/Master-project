from collections import defaultdict
import os
import glob
import json
import tqdm
import numpy as np
import cv2


# datacollection
def check_file(filename, allow_pickle=True):
    try:
        data = np.load(filename, allow_pickle=allow_pickle)
        if(len(data['rgb_static'].shape) != 3 or
                len(data['rgb_gripper'].shape) != 3):
            raise Exception("Corrupt data")
    except Exception as e:
        # print(e)
        data = None
    return data


# Merge datasets using json files
def merge_datasets(directory_list, output_dir):
    new_data = {"train": {}, "validation": {}}
    episode_it = [0, 0]
    for dir in directory_list:
        json_path = os.path.join(dir, "episodes_split.json")
        with open(json_path) as f:
            data = json.load(f)

        # TRAINING EPISODES
        # Rename episode numbers if repeated
        data_split = ["train", "validation"]
        for split, episode in zip(data_split, episode_it):
            dataset_name = os.path.basename(os.path.normpath(dir))
            for key in data[split].keys():
                new_data[split]["/%s/%s" % (dataset_name, key)] = \
                    data[split][key]
                episode += 1
    # Write output
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    with open(output_dir + '/episodes_split.json', 'w') as outfile:
        json.dump(new_data, outfile, indent=2)


# Select files that have a segmentation mask
def select_files(episode_files, remove_blank_masks):
    # Skip first n files for static cams since
    # affordances are incomplete at the beginning of episodes
    data = []
    for file in tqdm.tqdm(episode_files):
        head, tail = os.path.split(file)
        # Remove extension name
        file_name = tail.split('.')[0]
        # Last folder
        file_relative_path = os.path.basename(os.path.normpath(head))
        file_name = os.path.join(file_relative_path, file_name)
        if(remove_blank_masks):
            np_file = np.load(file)
            mask = np_file["mask"]  # (H, W)
            gripper_label = "gripper_cam" in file_name and mask.max() > 0
            enough_labels = "static_cam" in file_name \
                and len(np_file['centers']) > 5
            # Only add imgs where gripper is almost completely closed
            # at least one pixel is not background
            if(gripper_label or enough_labels):
                data.append(file_name)
        else:
            data.append(file_name)
    return data


def split_by_ep(root_dir, remove_blank_mask_instances=False):
    data = {"train": [], "validation": []}
    # Episodes are subdirectories
    n_episodes = 0
    if(isinstance(root_dir, list)):
        for dir_i in root_dir:
            n_episodes += len(glob.glob(dir_i + "/*/"))
    n_episodes = len(glob.glob(root_dir + "/*/"))
    n_val_ep = n_episodes//4
    val_ep = [i for i in range(n_episodes - n_val_ep, n_episodes)]
    train_ep = [ep for ep in range(n_episodes) if ep not in val_ep]
    data["validation"] = {"episode_%02d" % e: [] for e in val_ep}
    data["train"] = {"episode_%02d" % e: [] for e in train_ep}
    skip_first_frames = False

    for ep in tqdm.tqdm(range(n_episodes)):
        ep_dir = os.path.join(root_dir, "episode_%02d" % ep)
        ep_str = 'episode_%02d' % ep
        split = "validation" if ep in val_ep else "train"

        gripper_cam_files = glob.glob("%s/data/*/*gripper*" % ep_dir)
        selected_gripper_files = select_files(gripper_cam_files,
                                              remove_blank_mask_instances)

        static_cam_files = glob.glob("%s/data/*/*static*" % ep_dir)
        selected_static_files = select_files(static_cam_files,
                                             remove_blank_mask_instances)
        data[split].update({ep_str: selected_gripper_files})
        data[split][ep_str].extend(selected_static_files)
    return data


def split_by_ts(root_dir, remove_blank_mask_instances=False):
    data = {"train": [], "validation": []}

    # Count episodes
    n_episodes = 0
    if(isinstance(root_dir, list)):
        for dir_i in root_dir:
            n_episodes += len(glob.glob(dir_i + "/*/"))
    n_episodes = len(glob.glob(root_dir + "/*/"))
    data["validation"] = {"episode_%02d" % e: [] for e in range(n_episodes)}
    data["train"] = {"episode_%02d" % e: [] for e in range(n_episodes)}
    # Iterate episodes
    for ep in tqdm.tqdm(range(n_episodes)):
        ep_dir = os.path.join(root_dir, "episode_%02d" % ep)
        full_gripper_cam_files = glob.glob("%s/data/*/*gripper*" % ep_dir)

        full_static_cam_files = glob.glob("%s/data/*/*static*" % ep_dir)
        gripper_data = select_files(full_gripper_cam_files,
                                    remove_blank_mask_instances)

        static_data = select_files(full_static_cam_files,
                                   remove_blank_mask_instances)

        for split in ["validation", "train"]:
            ep_str = 'episode_%02d' % ep
            if(split == "validation"):
                gripper_cam_files = gripper_data[-len(gripper_data)//4:]
                static_cam_files = static_data[-len(static_data)//4:]
            else:
                gripper_cam_files = gripper_data[:-len(gripper_data)//4]
                static_cam_files = static_data[:-len(static_data)//4]

            data[split].update({ep_str: gripper_cam_files})
            data[split][ep_str].extend(static_cam_files)

    return data


# Split episodes into train and validation
def create_data_ep_split(root_dir, label_by_ep,
                         remove_blank_mask_instances=True):
    if(label_by_ep):
        data = split_by_ep(root_dir, remove_blank_mask_instances)
    else:
        data = split_by_ts(root_dir, remove_blank_mask_instances)
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
def save_data(data_dict, directory, sub_dir, save_viz=False):
    if(save_viz):
        data_dir, viz_dir, viz_frames, viz_aff = \
            create_dirs(directory, sub_dir, ['data',
                                             'viz_direction',
                                             'viz_frames',
                                             'viz_affordance'])
    else:
        data_dir = create_dirs(directory, sub_dir, ['data'])[0]
    for img_id, img_dict in data_dict.items():
        # Write vizualization output
        if(save_viz):
            aff_viz_filname = os.path.join(viz_aff, img_id) + ".jpg"
            dir_viz_filname = os.path.join(viz_dir, img_id) + ".jpg"
            frame_viz_filname = os.path.join(viz_frames, img_id) + ".jpg"

            affordance = img_dict['viz_out']
            directions = img_dict['viz_dir']
            orig_frame = img_dict['frame']

            cv2.imwrite(aff_viz_filname, affordance)
            cv2.imwrite(dir_viz_filname, directions)
            cv2.imwrite(frame_viz_filname, orig_frame)
            img_dict.pop('viz_out')
            img_dict.pop('viz_dir')

        # img_dict = {"frame":np.array, "mask":np.array, "centers": np.array}
        # frame is in BGR
        filename = os.path.join(data_dir, img_id) + ".npz"
        np.savez_compressed(
            filename,
            **img_dict
        )


# Ger valid numpy files with raw data
def get_files(path, extension, recursive=False):
    if(not os.path.isdir(path)):
        print("path does not exist: %s" % path)
    search_str = "/*.%s" % extension if not recursive else "**/*.%s" % extension
    files = glob.glob(path + search_str)
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
