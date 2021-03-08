import os
import glob
import json
import tqdm
import shutil


def split_episodes(all_files, masks_dir, ep_start_end_ids,
                   remove_blank_mask_instances, data):
    all_files.sort()
    end_ids = ep_start_end_ids[:, -1]
    start_ids = ep_start_end_ids[:, 0]
    e = 0
    for file in tqdm.tqdm(all_files):
        head, tail = os.path.split(file)
        frame_id = int(tail.split('_')[-1][:-4])

        # Episode
        if(frame_id >= start_ids[e] and frame_id <= end_ids[e]):
            file_relative_path = head.replace(masks_dir, "")
            file_name = tail.split('.')[0]  # Remove extension name
            file_name = os.path.join(file_relative_path, file_name)
            if(remove_blank_mask_instances):
                mask = np.load(file)  # (H, W)
                if(mask.max() > 0):  # at least one pixel is not background
                    data['episode_%d' % e].append(file_name)
            else:
                data['episode_%d' % e].append(file_name)
        else:
            e += 1
    return data


# File to get segmentation maks, and save data
def separate_validation_frames(play_data_dir, save_dir, n_eval_ep):
    frames_dir = save_dir + "/frames"
    gripper_cam_files = glob.glob(frames_dir + "/*/*gripper*")
    static_cam_files = glob.glob(frames_dir + "/*/*static*")
    data = {"gripper": gripper_cam_files,
            "static": static_cam_files}
    # Split data
    ep_start_end_ids = np.load(os.path.join(
        play_data_dir,
        "ep_start_end_ids.npy"))

    # Evaluation episodes always the last
    start_id = ep_start_end_ids[:, 0][- n_eval_ep]
    end_id = ep_start_end_ids[:, -1][-1]

    for cam, cam_files in data.items():
        os.makedirs("%s/validation/%s/" % (save_dir, cam))
        for file in tqdm.tqdm(cam_files):
            head, tail = os.path.split(file)
            frame_id = int(tail.split('_')[-1][:-4])

            if(frame_id >= start_id and frame_id <= end_id):
                shutil.copy(file,
                            "%s/validation/%s/%s" % (save_dir, cam, tail))


# File to get segmentation maks, and save data
def create_data_ep_split(play_data_dir, root_dir,
                         remove_blank_mask_instances=True):
    masks_dir = root_dir + "/masks"
    gripper_cam_files = glob.glob(masks_dir + "/*/*gripper*")
    scatic_cam_files = glob.glob(masks_dir + "/*/*static*")

    # Split data
    ep_start_end_ids = np.load(os.path.join(
        play_data_dir,
        "ep_start_end_ids.npy"))

    data = {'episode_%d' % e: [] for e in range(len(ep_start_end_ids))}
    data = split_episodes(gripper_cam_files, masks_dir, ep_start_end_ids,
                          remove_blank_mask_instances, data)
    data = split_episodes(scatic_cam_files, masks_dir, ep_start_end_ids,
                          remove_blank_mask_instances, data)

    with open(root_dir+'/ep_data.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)


def create_data_split(root_dir, remove_blank_mask_instances=True):
    data = {'train': [], "validation": []}
    masks_dir = root_dir + "/masks"
    all_files = glob.glob(masks_dir + "/*/*")

    valid_frames = []
    if (remove_blank_mask_instances):
        for file in tqdm.tqdm(all_files):
            mask = np.load(file)  # (H, W)
            if(mask.max() > 0):
                # at least one pixel is not background
                valid_frames.append(file)
    else:
        valid_frames = all_files

    # Split data
    val_idx = np.random.choice(
                len(valid_frames),
                len(valid_frames)//3,
                replace=False)

    for idx, file in tqdm.tqdm(enumerate(valid_frames)):
        head, tail = os.path.split(file)
        # Only keep subdirectories of masks
        file_relative_path = head.replace(masks_dir, "")
        file_name = tail.split('.')[0]  # Remove extension name
        relative_path = os.path.join(file_relative_path, file_name)
        if(idx in val_idx):  # Validation
            data['validation'].append(relative_path)
        else:
            data['train'].append(relative_path)

    with open(root_dir+'/data.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)


def create_dirs(root_dir, sub_dir, directory_lst):
    dir_lst = [root_dir]
    for d_name in directory_lst:
        dir_lst.append(root_dir + "/%s/%s/" % (d_name, sub_dir))

    for directory in dir_lst:
        if(not os.path.exists(directory)):
            os.makedirs(directory)
    dir_lst.pop(0)  # Remove root_dir
    return dir_lst


def save_data(data_dict, directory, sub_dir, save_viz=True):
    if(save_viz):
        frames_dir, masks_dir, viz_out_dir = \
            create_dirs(directory, sub_dir, ['frames', 'masks', 'viz_out'])
    else:
        frames_dir, masks_dir = \
            create_dirs(directory, sub_dir, ['frames', 'masks'])
    for img_id, img_dict in data_dict.items():
        filename = img_id
        # Write original image
        img_filename = os.path.join(frames_dir, filename) + ".jpg"
        cv2.imwrite(img_filename, img_dict['frame'])  # Save images
        # Write vizualization output
        if(save_viz):
            img_filename = os.path.join(viz_out_dir, filename) + ".jpg"
            cv2.imwrite(img_filename, img_dict['viz_out'])  # Save images

        mask_filename = os.path.join(masks_dir, filename) + ".npy"
        with open(mask_filename, 'wb') as f:  # Save masks
            np.save(f, img_dict['mask'])


def get_files(path, extension):
    if(not os.path.isdir(path)):
        print("path does not exist: %s" % path)
    files = glob.glob(path + "/*.%s" % extension)
    if not files:
        print("No *.%s files found in %s" % (extension, path))
    files.sort()
    return files