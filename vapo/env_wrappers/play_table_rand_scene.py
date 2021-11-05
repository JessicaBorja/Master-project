import logging
from omegaconf import OmegaConf
from random import shuffle
from collections import defaultdict
import numpy as np
from vr_env.scene.play_table_scene import PlayTableScene
logger = logging.getLogger(__name__)


class PlayTableRandScene(PlayTableScene):
    def __init__(self, **args):
        super(PlayTableRandScene, self).__init__(**args)
        # all the objects
        self.obj_names = list(self.object_cfg['movable_objects'].keys())
        self.objs_per_class, self.class_per_obj = {}, {}
        self._find_obj_class()
        if('positions' in args):
            self.rand_positions = args['positions']
        else:
            self.rand_positions = None

        # Load Environment
        self.target = "banana"
        if(self.rand_positions):
            self.load_rand_scene(load_scene=True)
        else:
            self.table_objs = self.obj_names
            self.pick_rand_obj()

    # Loading random objects
    def _find_obj_class(self):
        obj_names = self.obj_names
        objs_per_class = defaultdict(list)
        class_per_object = {}

        # Find classes in objects
        for name in obj_names:
            if("_" in name):
                # Remove last substring (i.e. class count)
                class_name = "_".join(name.split('_')[:-1])
                objs_per_class[class_name].append(name)
            else:
                objs_per_class["misc"].append(name)

        # Assign objects to classes
        classes = list(objs_per_class.keys())
        for class_name in classes:
            objs_in_class = objs_per_class[class_name]
            if len(objs_in_class) <= 1:
                objs_per_class["misc"].extend(objs_in_class)
                class_per_object.update({obj: "misc" for obj in objs_in_class})
                objs_per_class.pop(class_name)
            else:
                class_per_object.update(
                    {obj: class_name for obj in objs_in_class})
        self.objs_per_class = objs_per_class
        self.class_per_obj = class_per_object

    def pick_rand_obj(self, p_dist=None):
        '''
            p_dist = {class: counts}
        '''
        if(p_dist):
            labels = list(p_dist.keys())
            counts = list(p_dist.values())
            weights = np.array(counts)
            w_sum = weights.sum(axis=0)
            if(w_sum == 0):
                weights = np.ones_like(weights) * (1 / len(weights))
            else:
                # Make less successful more likely
                weights = weights / w_sum  # Normalize to sum 1
                weights = 1 - weights
                weights = weights / weights.sum(axis=0)
            choose_class = self.np_random.choice(labels, p=weights)
            _class_in_table = []
            for obj in self.table_objs:
                if(self.class_per_obj[obj] == choose_class):
                    _class_in_table.append(obj)
            self.target = self.np_random.choice(_class_in_table)
        else:
            self.target = self.np_random.choice(self.table_objs)

    def load_scene_with_objects(self, obj_lst, load_scene=False):
        '''
            obj_lst: list of strings containing names of objs
            load_scene: Only true in initialization of environment
        '''
        assert len(obj_lst) <= len(self.rand_positions)
        rand_pos = self.rand_positions[:len(obj_lst)]
        shuffle(obj_lst)
        # movable_objs is a reference to self.object_cfg
        if(load_scene):
            movable_objs = self.object_cfg['movable_objects']
            # Add positions to new table ojects
            for name, new_pos in zip(obj_lst, rand_pos):
                movable_objs[name]["initial_pos"][:2] = new_pos

            # Add other objects away from view
            far_objs = {k: v for k, v in movable_objs.items()
                        if k not in obj_lst}
            far_pos = [[100 + 20 * i, 0] for i in range(len(far_objs))]
            for i, (name, properties) in enumerate(far_objs.items()):
                movable_objs[name]["initial_pos"][:2] = far_pos[i]
            self.load()
        else:
            movable_objs = {obj.name: i for i, obj in
                            enumerate(self.movable_objects)}
            # Add positions to new table ojects
            for name, new_pos in zip(obj_lst, rand_pos):
                _obj = self.movable_objects[movable_objs[name]]
                _obj.initial_pos[:2] = new_pos

            # Add other objects away from view
            far_objs = {k: v for k, v in movable_objs.items()
                        if k not in obj_lst}
            far_pos = [[100 + 20 * i, 0] for i in range(len(far_objs))]
            for i, (name, properties) in enumerate(far_objs.items()):
                _obj = self.movable_objects[movable_objs[name]]
                _obj.initial_pos[:2] = far_pos[i]

        self.table_objs = obj_lst.copy()
        self.reset()
        self.pick_rand_obj()

    def load_rand_scene(self, replace_objs=None, load_scene=False):
        n_objs = len(self.rand_positions)
        if(replace_objs):
            # Replace some objs
            rand_objs = {o for o in self.table_objs
                         if o not in replace_objs}
            for obj in replace_objs:
                obj_class = self.class_per_obj[obj]
                # Replace obj for another of the same class
                remaining_objs = [o for o in self.objs_per_class[obj_class]
                                  if o not in rand_objs and o != obj]
                # Replace new obj
                if(len(remaining_objs) > 0):
                    rand_obj = self.np_random.choice(remaining_objs)
                    rand_objs.add(rand_obj)
                else:
                    continue
            n_objs -= len(rand_objs)
            rand_objs = list(rand_objs)
            choose_from = [o for o in self.obj_names
                           if o not in rand_objs]
            rand_objs.extend(self.np_random.choice(choose_from, n_objs))
            print_str = "Classes in env:"
            for obj in rand_objs:
                print_str += "%s: %s \n" % (obj, self.class_per_obj[obj])
            logger.info(print_str)
        else:
            # Full random scene
            choose_from = []
            for v in self.objs_per_class.values():
                choose_from.extend(v)

            # At least 1 obj from each class in env
            rand_objs = []
            rand_obj_classes = list(self.objs_per_class.keys())
            for class_name in rand_obj_classes:
                class_objs = self.objs_per_class[class_name]
                rand_obj = self.np_random.choice(class_objs, 1)
                rand_objs.extend(rand_obj)
                choose_from.remove(rand_obj)
                n_objs -= 1
            rand_objs.extend(self.np_random.choice(choose_from,
                                                   n_objs,
                                                   replace=False))
        self.load_scene_with_objects(rand_objs, load_scene=load_scene)
