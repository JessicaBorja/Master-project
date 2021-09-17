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
        self.global_obs_it = 0
        self.aff_net_static_cam = self._init_static_cam_aff_net(aff_cfg)
        self.class_label = class_label
        if(env.task == "pickup"):
            self.box_mask, self.box_3D_end_points = self.get_box_pos_mask(self.env)

    def compute(self, env=None,
                return_all_centers=False,
                p_dist=None,
                noisy=False):
        if(env is None):
            env = self.env
        if(self.mode == "affordance"):
            res = self._compute_target_aff(env)
            target_pos, no_target, object_centers = res
            # if env.task == "slide" or env.task == "hinge":
            #     # Because it most likely will detect the door and not the handle
            #     # target_pos = np.array([target_pos[0],
            #     #                        target_pos[1] - 0.075,
            #     #                        target_pos[2]])
            if(noisy):
                target_pos += np.random.normal(loc=0,
                                               scale=[0.01, 0.01, 0.005],
                                               size=(len(target_pos)))
            if(return_all_centers):
                obj_centers = []
                for center in object_centers:
                    obj = {}
                    obj["target_pos"] = center
                    obj["target_str"] = \
                        self.find_env_target(env, center)
                    obj_centers.append(obj)
                res = (target_pos, no_target, obj_centers)
            else:
                res = (target_pos, no_target)
                if(env.task == "pickup"):
                    env_target = self.find_env_target(env, target_pos)
                    env.target = env_target
                    env.unwrapped.target = env_target
        else:
            if(p_dist):
                env.pick_rand_obj(p_dist)
            res = self._env_compute_target(env, noisy)
        return res

    # Env real target pos
    def _env_compute_target(self, env=None, noisy=False):
        if(not env):
            env = self.env
        # This should come from static cam affordance later on
        target_pos, _ = env.get_target_pos()
        # 2 cm deviation
        target_pos = np.array(target_pos)
        if(noisy):
            target_pos += np.random.normal(loc=0, scale=[0.005, 0.005, 0.01],
                                        size=(len(target_pos)))

        # always returns a target position
        no_target = False
        return target_pos, no_target

    # Aff-center model
    def _compute_target_aff(self, env=None):
        if(not env):
            env = self.env
        # Get environment observation
        cam = env.cameras[self.cam_id]
        obs = env.get_obs()
        depth_obs = obs["depth_obs"][self.cam_id]
        orig_img = obs["rgb_obs"][self.cam_id]

        # Apply validation transforms
        img_obs = torch.tensor(orig_img).permute(2, 0, 1).unsqueeze(0).cuda()
        img_obs = self.aff_transforms(img_obs)

        # Predict affordances and centers
        _, aff_probs, aff_mask, center_dir = \
            self.aff_net_static_cam.forward(img_obs)
        if(env.task == "pickup"):
            aff_mask = aff_mask - self.box_mask

        # Filter by class
        if(self.class_label is not None):
            class_mask = torch.zeros_like(aff_mask)
            class_mask[aff_mask == self.class_label] = 1
        else:
            class_mask = aff_mask

        aff_mask, center_dir, object_centers, object_masks = \
            self.aff_net_static_cam.predict(class_mask, center_dir)

        # Visualize predictions
        if(env.viz or env.save_images):
            img_dict = viz_aff_centers_preds(orig_img, aff_mask, aff_probs,
                                             center_dir, object_centers,
                                             object_masks, "static",
                                             self.global_obs_it,
                                             self.env.save_images,
                                             resize=(300, 300))
            for img_path, img in img_dict.items():
                folder = os.path.dirname(img_path)
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(img_path, img)
            self.global_obs_it += 1

        # To numpy
        aff_probs = torch_to_numpy(aff_probs[0].permute(1, 2, 0))  # H, W, 2
        object_masks = torch_to_numpy(object_masks[0])  # H, W

        # Plot different objects
        no_target = False
        if(len(object_centers) > 0):
            target_px = object_centers[0]
        else:
            # No center detected
            default = self.initial_pos
            no_target = True
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

        # World coords
        world_pts = []
        for o in object_centers:
            x = o.detach().cpu().numpy()
            x = (x * orig_shape / pred_shape).astype("int64")
            if(env.task == "drawer" or env.task == "slide"):
                # As center might  not be exactly in handle
                # look for max depth around neighborhood
                n = 10
                depth_window = depth_obs[x[0] - n:x[0] + n, x[1] - n:x[1] + n]
                proposal = np.argwhere(depth_window == np.min(depth_window))[0]
                v = x[0] - n + proposal[0]
                u = x[1] - n + proposal[1]
            else:
                v, u = x
            world_pt = np.array(cam.deproject([u, v], depth_obs))
            world_pts.append(world_pt)

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
            # os.makedirs("./static_centers/", exist_ok=True)
            # cv2.imwrite("./static_centers/img_%04d.jpg" % self.global_obs_it,
            #             out_img)

        # p.addUserDebugText("t",
        #                    textPosition=target_pos,
        #                    textColorRGB=[0, 1, 0])
        return target_pos, no_target, world_pts

    def find_env_target(self, env, target_pos):
        min_dist = np.inf
        env_target = env.target
        for name in env.table_objs:
            target_obj = env.objects[name]
            base_pos = p.getBasePositionAndOrientation(target_obj["uid"],
                                                       physicsClientId=env.cid)[0]
            if(p.getNumJoints(target_obj["uid"]) == 0):
                pos = base_pos
            else:  # Grasp link
                pos = p.getLinkState(target_obj["uid"], 0)[0]
            dist = np.linalg.norm(pos - target_pos)
            if(dist < min_dist):
                env_target = name
                min_dist = dist
        return env_target

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
