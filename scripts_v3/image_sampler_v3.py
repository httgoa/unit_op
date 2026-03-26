import random
import numpy as np
import os
import math
from typing import List, Tuple, Optional, Dict, Any
from target_sampler_v3 import SingleColoredTarget

class ImageSampler():
    def __init__(self, image_size, bg_color):
        self.image_size = image_size
        self.image_height, self.image_width = image_size
        self.bg_color = bg_color
        self.image_array = np.full((self.image_height, self.image_width), self.bg_color, dtype=np.int64)
        self.targets = {}
    
    def add_target(self, target:SingleColoredTarget, target_coords):
        # precondition: target_pos should be a valid tuple representing coords
        th, tw = target.target_array.shape
        ttop, tleft = target_coords
        self.image_array[ttop:ttop+th, tleft:tleft+tw] = target.target_array.copy()
        pos_id = self.coords2pos(target_coords)
        self.targets[pos_id] = target
        
    def coords2pos(self, coords):
        top, left = coords
        return top * self.image_width + left
    
    def pos2coords(self, pos):
        top = pos // self.image_width
        left = pos % self.image_width
        return (top, left)
    
    def num_targets(self):
        return len(self.targets)

    def get_all_targets(self):
        # in ascending order of position
        targets = [value for key, value in sorted(list(self.targets.items()), key=lambda item:item[0])]
        return targets
    
    def get_ith_target(self, i):
        sorted_targets = list(sorted(list(self.targets.items()), key=lambda item:item[0]))
        ith = sorted_targets[i]
        i_pos, i_target = ith
        return i_pos, i_target
    
    def update_target(self, target, pos_id):
        # update index
        self.targets[pos_id] = target
        # update array
        coords = self.pos2coords(pos_id)
        top, left = coords
        th, tw = target.target_array.shape
        self.image_array[top:top+th, left:left+tw] = target.target_array.copy()
        
    
    # can add other functions such as the leftmost target, the largest target, etc.
    # add them when you are preparing multi-target data
    
    
    