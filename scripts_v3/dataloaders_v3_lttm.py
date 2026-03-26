import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
import random
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List

# Add path to scripts directory to use Target and unit operation functions
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'scripts_v3')
sys.path.append(scripts_dir)
from unit_op_und_funcs_v3 import UnitOpUndFuncSampler
from unit_op_gen_funcs_v3 import UnitOpGenFuncSampler
from target_sampler_v3 import SingleColoredTarget
from image_sampler_v3 import ImageSampler



class UnitOpDataset(Dataset):
    """
    Dataset for training STL (Structural Transformation Learning) and LTTM (Latent Transformation Transfer Module).
    For each sample, it provides a tuple of (ia, ib, op_name, op_arg) where ib is derived from applying a unit operation to ia.

    The target object only occupies a portion of the full image and is randomly positioned within it.

    Samples a variety of different transformations from unit_op_und_funcs and unit_op_gen_funcs,
    with diverse parameters to increase data variety.
    """

    def __init__(self, num_samples=10000, num_colors=11, image_size=(64, 64), target_size_range=(10, 40), seed=None):
        """
        Initialize the STL dataset.

        Args:
            num_samples (int): Number of (ia, ib, op_name, op_arg) samples to generate
            num_colors (int): Number of possible color values
            image_size (tuple): Size of the full 2D grid (height, width)
            target_size_range (tuple): Range of possible target sizes (min, max)
            seed (int, optional): Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_colors = num_colors
        self.image_size = image_size
        self.target_size_range = target_size_range

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize the function samplers
        self.und_func_sampler = UnitOpUndFuncSampler()
        self.gen_func_sampler = UnitOpGenFuncSampler()

        # Generate the dataset
        self.data = self._generate_dataset()
        
    def _generate_image(self, image_size):
        bg_color = random.choice(list(range(0,self.num_colors-1))) # self.num_colors-1 is reserved for labeling
        ih, iw = image_size
        image_object = ImageSampler(image_size, bg_color)
        return image_object
    
    def _check_different_targets(self, target_ori:SingleColoredTarget, target_trans:SingleColoredTarget):
        array_ori = target_ori.target_array
        array_trans = target_trans.target_array
        return not np.array_equal(array_ori, array_trans)
    
    def _sample_func(self):
        func_type = random.choice(["und", "gen"])
        if func_type == "und":
            op_name, op_arg = self.und_func_sampler.get_func()
        elif func_type == "gen":
            op_name, op_arg = self.gen_func_sampler.get_func()
        return (func_type, op_name, op_arg)
    
    def _apply_func(self, target:SingleColoredTarget, func_type:str, op_name:str, op_arg:str):
        assert func_type == "und" or "gen"
        if func_type == "und":
            new_target = self.und_func_sampler.apply_func(target, op_name, op_arg)
        elif func_type == "gen":
            new_target = self.gen_func_sampler.apply_func(target, op_name, op_arg)
        return new_target
    
    def _generate_target_pairs(self, func_type, op_name, op_arg, bg_color, target_cnt, bbox_min, bbox_max):
        change_cnt, total_cnt = 0, 0
        changed_pairs = []
        unchanged_pairs = []
        loop_cnt =0
        
        def eval_continue_condition():
            if func_type == "und":
                continue_condition = len(changed_pairs) < 0.3 * target_cnt or len(unchanged_pairs) < 0.3 * target_cnt or total_cnt < target_cnt
            elif func_type == "gen":
                continue_condition = len(changed_pairs) < 0.3 * target_cnt or total_cnt < target_cnt
            return continue_condition

        while eval_continue_condition():
            loop_cnt += 1
            success = False
            while not success:
                if loop_cnt > target_cnt*2:
                    print(f"target_cnt={target_cnt}. loop cnt {loop_cnt} for func {op_name} and {op_arg}. change_cnt={change_cnt} and total_cnt={total_cnt}, len of changed pairs {len(changed_pairs)}, len of unchanged pairs {len(unchanged_pairs)}")
                    # hard code
                    if op_name == "check_repetition" and op_arg!="none":
                        target_ori = self._create_random_target(bbox_min=bbox_min, bbox_max=bbox_max, bg_color=bg_color, repetitive=True, filled=False)
                    elif op_name == "check_symmetric" and op_arg!="none":
                        target_ori = self._create_random_target(bbox_min=bbox_min, bbox_max=bbox_max, bg_color=bg_color, symmetric=True, filled=False)
                    else:
                        target_ori = self._create_random_target(bbox_min=bbox_min, bbox_max=bbox_max, bg_color=bg_color)
                else:
                    target_ori = self._create_random_target(bbox_min=bbox_min, bbox_max=bbox_max, bg_color=bg_color)
                    
                th, tw = target_ori.target_array.shape
                success = th<=bbox_max and th>=bbox_min and tw<=bbox_max and tw>=bbox_min
            # transform target
            target_trans = self._apply_func(target_ori, func_type, op_name, op_arg)

            # check if transformed target is different from the original target
            different = self._check_different_targets(target_ori, target_trans)
            if different:
                change_cnt += 1
            total_cnt += 1
            
            # append target pairs to corresponding list
            if different:
                changed_pairs.append((target_ori, target_trans))
            else:
                unchanged_pairs.append((target_ori, target_trans))
        
        # select a total number of target_cnt pairs with relatively balanced pos,neg samples
        result = []
        changed_selected, unchanged_selected = 0, 0
        while changed_selected + unchanged_selected < target_cnt:
            if changed_selected >= unchanged_selected:
                if unchanged_selected < len(unchanged_pairs):
                    result.append(unchanged_pairs[unchanged_selected])
                    unchanged_selected += 1
                else:
                    # add all other changed pairs 
                    result.extend(changed_pairs[changed_selected:target_cnt-unchanged_selected])
                    changed_selected += target_cnt - unchanged_selected - changed_selected
            else:
                if changed_selected < len(changed_pairs):
                    result.append(changed_pairs[changed_selected])
                    changed_selected += 1
                else:
                    # all all other unchanged pairs
                    result.extend(unchanged_pairs[unchanged_selected:target_cnt-changed_selected])
                    unchanged_selected += target_cnt - unchanged_selected - changed_selected
        
        random.shuffle(result)
        return result
            
    def _update_targets_to_images(self, target_pairs, image_ori, image_trans):
        ih, iw = image_ori.image_size
        for i in range(len(target_pairs)):
            # len(target_pairs)=4
            grid_row = i // 2
            grid_col = i % 2
            coords = (grid_row * (ih // 2)+1, grid_col * (iw // 2)+1) # offset 1 so that objects are not on the edges
            target_ori, target_trans = target_pairs[i]
            image_ori.add_target(target_ori, coords)
            image_trans.add_target(target_trans, coords)
        return

    def _generate_dataset(self):
        """Generate the dataset of (ia, ib, op_name, op_arg) tuples."""
        dataset = []

        for i in range(self.num_samples):
            if i % 10 == 0:
                print(f"Generating sample {i}/{self.num_samples}")
            
            # sample a func
            func_type, op_name, op_arg = self._sample_func()
            if func_type == "und":
                target_cnt = 4
                bbox_min = 20
                bbox_max = 30
            elif func_type == "gen":
                target_cnt = 4
                bbox_min = 20
                bbox_max = 30
                
            # generate images a, b
            imo_a = self._generate_image(self.image_size)
            imo_b = deepcopy(imo_a)
            # generate targets for a, b
            tab_pairs = self._generate_target_pairs(func_type, op_name, op_arg, imo_a.bg_color, target_cnt, bbox_min, bbox_max)
            # update target to images
            self._update_targets_to_images(tab_pairs, imo_a, imo_b)
            
            # generate images c, d
            imo_c = self._generate_image(self.image_size)
            imo_d = deepcopy(imo_c)
            # generate targets for c, d
            tcd_pairs = self._generate_target_pairs(func_type, op_name, op_arg, imo_c.bg_color, target_cnt, bbox_min, bbox_max)
            # update target to images
            self._update_targets_to_images(tcd_pairs, imo_c, imo_d)
            
            # Convert to tensors and apply one-hot encoding
            ia_tensor = self._preprocess_array(imo_a.image_array)
            ib_tensor = self._preprocess_array(imo_b.image_array)
            ic_tensor = self._preprocess_array(imo_c.image_array)
            id_tensor = self._preprocess_array(imo_d.image_array)

            # Add to dataset with operation name and argument
            data_item = {
                "a":ia_tensor,
                "b":ib_tensor,
                "c":ic_tensor,
                "d":id_tensor,
                "op_name":op_name,
                "op_arg":op_arg
            }
            dataset.append(data_item)

        return dataset
    
    def _create_random_target(self, bbox_min, bbox_max, bg_color, symmetric=None, repetitive=None, filled=None):
        # Create the target
        success = False
        # symmetric and repetitive could not be True at the same time
        if symmetric is None:
            symmetric = random.choice([True, False])
        if repetitive is None:
            repetitive = random.choice([True, False])
        if symmetric and repetitive:
            symmetric = False
            repetitive = False
        if filled is None:
            filled = random.choice([True, False])
        while not success:
            target = SingleColoredTarget(
                bbox_min,
                bbox_max,
                bg_color=bg_color,
                filled=filled,
                symmetric=symmetric,
                repetitive=repetitive
            )
            unique_colors_list = list(np.unique(target.target_array))
            fg_colors_list = [c for c in unique_colors_list if c!=bg_color]
            success = len(fg_colors_list) >= 1
            if not success:
                print(f"failed sampling case: fg_colors_list {fg_colors_list} and bg_color={bg_color} ")
        #print(f"number of unique different values {len(list(np.unique(target.target_array)))}")
        return target
        

    def _preprocess_array(self, arr):
        """Convert array to tensor with one-hot encoding."""
        # Convert to PyTorch tensor
        # Make a copy to ensure positive strides
        arr = np.ascontiguousarray(arr)
        arr_tensor = torch.from_numpy(arr).long()

        # One-hot encoding
        arr_onehot = F.one_hot(arr_tensor, num_classes=self.num_colors)

        # Permute to [num_colors, H, W] format
        arr_onehot = arr_onehot.permute(2, 0, 1).float()

        return arr_onehot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]