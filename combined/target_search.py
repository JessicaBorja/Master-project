import numpy as np
from utils.img_utils import torch_to_numpy, viz_aff_centers_preds
from affordance_model.segmentator_centers import Segmentator
from omegaconf import OmegaConf
import os
import cv2
import torch
import pybullet as p


class TargetSearch():
    def __init__(self, env, mode,
                 aff_transforms=None, aff_cfg=None, class_label=None,
                 cam_id=0, initial_pos=None, rand_target=False) -> None:
        self.env = env
        self.mode = mode
        self.uniform_sample = False
        self.cam_id = cam_id
        self.random_target = rand_target
        self.initial_pos = initial_pos
        self.aff_transforms = aff_transforms
        self.affordance_cfg = aff_cfg
        self.aff_net_static_cam = self._init_static_cam_aff_net(aff_cfg)
        self.static_cam_imgs = {}
        self.class_label = class_label
        self.box_mask = None

    def compute(self, env=None,
                global_it=0,
                return_all_centers=False,
                p_dist=None):
        if(env is None):
            env = self.env
        cam = env.env.env.camera_manager.static_cam  # static_cam
        orig_img, depth_img = cam.get_image()
        res = self._compute_target_aff(env, cam,
                                       depth_img,
                                       orig_img,
                                       global_it)
        if(not return_all_centers):
            res = res[:2]
        return res

    # Aff-center model
    def _compute_target_aff(self, env, cam, depth_obs, orig_img,
                            global_obs_it=0):

        # Apply validation transforms
        img_obs = torch.tensor(orig_img).permute(2, 0, 1).unsqueeze(0).cuda()
        img_obs = self.aff_transforms(img_obs)

        # Predict affordances and centers
        _, aff_probs, aff_mask, center_dir = \
            self.aff_net_static_cam.forward(img_obs)
        # if(env.task == "pickup" and self.box_mask is not None):
        #     aff_mask = aff_mask - self.box_mask

        # Filter by class
        if(self.class_label is not None):
            class_mask = torch.zeros_like(aff_mask)
            class_mask[aff_mask == self.class_label] = 1
        else:
            class_mask = aff_mask

        aff_mask, center_dir, object_centers, object_masks = \
            self.aff_net_static_cam.predict(class_mask, center_dir)

        # Visualize predictions
        # if(env.viz or env.save_images):
        img_dict = viz_aff_centers_preds(orig_img, aff_mask, aff_probs,
                                         center_dir, object_centers,
                                         object_masks, "static",
                                         global_obs_it,
                                         False)
            # self.static_cam_imgs.update(img_dict)
            #global_obs_it += 1

        # To numpy
        aff_probs = torch_to_numpy(aff_probs[0].permute(1, 2, 0))  # H, W, 2
        object_masks = torch_to_numpy(object_masks[0])  # H, W

        # No center detected
        no_target = len(object_centers) < 0
        if(no_target):
            default = self.initial_pos
            return np.array(default), no_target, []

        max_robustness = 0
        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class

        if(self.random_target):
            target_idx = np.random.randint(len(object_centers))
            # target_idx = object_centers[rand_target]
        else:
            # Look for most likely center
            for i, o in enumerate(object_centers):
                # Mean prob of being class 1 (foreground)
                robustness = np.mean(aff_probs[object_masks == obj_class[i], 1])
                if(robustness > max_robustness):
                    max_robustness = robustness
                    target_idx = i

        # Convert back to observation size
        pred_shape = aff_probs.shape[:2]
        orig_shape = depth_obs.shape[:2]

        T_world_cam = cam.get_extrinsic_calibration("panda")

        # World coords
        world_pts = []
        for o in object_centers:
            x = o.detach().cpu().numpy()
            x = (x * orig_shape / pred_shape).astype("int64")
            v, u = x
            pt_cam = cam.deproject([u, v], depth_obs, homogeneous=True)
            world_pt = T_world_cam @ pt_cam
            world_pts.append(world_pt[:3])

        # Recover target
        v, u = object_centers[target_idx]
        target_pos = world_pts[target_idx]
        if(self.env.save_images):
            out_img = cv2.drawMarker(np.array(orig_img[:, :, ::-1]),
                                     (u, v),
                                     (0, 255, 0),
                                     markerType=cv2.MARKER_CROSS,
                                     markerSize=12,
                                     line_type=cv2.LINE_AA)
            # cv2.imshow("out_img", out_img)
            # cv2.waitKey(1)
            cv2.imwrite("./static_centers/img_%04d.jpg" % self.global_obs_it,
                        out_img)

        return target_pos, no_target, world_pts

    def get_box_pos_mask(self, env):
        if(not env):
            env = self.env
        box_pos = env.objects["bin"]["initial_pos"]
        x, y, z = box_pos
        # Homogeneous cords
        box_top_left = [x - 0.12, y + 0.2, z, 1]
        box_bott_right = [x + 0.12, y - 0.2, z + 0.08, 1]

        # Static camera 
        cam = env.cameras[self.cam_id]

        u1, v1 = cam.project(np.array(box_top_left))
        u2, v2 = cam.project(np.array(box_bott_right))

        shape = (cam.width, cam.height)
        mask = np.zeros(shape, np.uint8)
        mask = cv2.rectangle(mask, (u1, v1), (u2, v2),
                             (1, 1, 1), thickness=-1)
        shape = (self.affordance_cfg.img_size, self.affordance_cfg.img_size)
        mask = cv2.resize(mask, shape)
        # cv2.imshow("box_mask", mask)
        # cv2.waitKey()

        # 1, H, W
        mask = torch.tensor(mask).unsqueeze(0).cuda()
        return mask, (box_top_left, box_bott_right)

    def _init_static_cam_aff_net(self, affordance_cfg):
        path = affordance_cfg.model_path
        aff_net = None
        if(os.path.exists(path)):
            hp = OmegaConf.to_container(affordance_cfg.hyperparameters)
            hp = OmegaConf.create(hp)
            aff_net = Segmentator.load_from_checkpoint(
                                path,
                                cfg=hp)
            aff_net.cuda()
            aff_net.eval()
            print("Static cam affordance model loaded (to find targets)")
        else:
            affordance_cfg = None
            path = os.path.abspath(path)
            raise TypeError(
                "target_search_aff.model_path does not exist: %s" % path)
        return aff_net
