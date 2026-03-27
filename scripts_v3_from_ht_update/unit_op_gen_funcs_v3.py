import os
import numpy as np
import random
from copy import deepcopy
from typing import Tuple
from target_sampler_v3 import SingleColoredTarget
from image_sampler_v3 import ImageSampler

######### unit op functions that perform some actions #########
#平移，翻转，旋转，缩放，取最小单元，重复
# TODO: after each operation, you should update relevant target properties!!!!

class UnitOpGenFuncSampler():
    def __init__(self):
        self.func_list = [
            "rotation",
            "flip",
            "delete",
            "fill",
            "color",
            "translation",
            "concatenate",
            "connect",
            "shoot",
            "hupscale",
            "vupscale",
            "hdownscale",
            "vdownscale",
        ]
        
        self.all_rot_degrees = [1, 2, 3]
        self.all_flip_axis = [0, 1]
        self.all_colors = list(range(0, 10))       # 0-9, 10 reserved for labeling
        self.all_scale_factors = [2, 3]            # integer upscale/downscale factors
        self.all_translation_offsets = [           # (dy, dx) pairs
            (-2, 0), (2, 0), (0, -2), (0, 2),
            (-1, 0), (1, 0), (0, -1), (0, 1),
        ]
        self.all_concat_directions = ["horizontal", "vertical"]
        self.all_shoot_directions = ["up", "down", "left", "right"]
        
    
    def rotation(self, target:SingleColoredTarget, op_arg:str=None):
        assert int(op_arg) in [1, 2, 3]
        rot90 = int(op_arg)
        new_target = deepcopy(target)
        new_array = np.rot90(new_target.target_array, k=rot90)
        new_target.target_array = new_array
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target
        
    def flip(self, target:SingleColoredTarget, op_arg:str=None):
        assert int(op_arg) in [0, 1]
        axis = int(op_arg)
        new_target = deepcopy(target)
        new_array = np.flip(new_target.target_array, axis=axis)
        new_target.target_array = new_array
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target
        
    def delete(self, target:SingleColoredTarget, op_arg:str=None):
        new_target = deepcopy(target)
        new_array = new_target.target_array
        new_array[new_array!=new_target.bg_color] = new_target.bg_color
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target
    
    # should also implement filling some part of the holes of the target
    def fill(self, target:SingleColoredTarget, op_arg:str=None):
        new_target = deepcopy(target)
        if target.hole_list:
            for hole in target.hole_list:
                for point in hole:
                    new_target.target_array[point] = new_target.fg_color
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target
    
    ##### other gen ops from ht
    def color(self, target:SingleColoredTarget, op_arg:int=None):
        # change fg_color pixels of the target to op_arg color
        # works for both:
        #   - original target: pixels == fg_color are all fg pixels
        #   - label_pixels result: pixels == fg_color are the labeled subset, rest is LABEL_COLOR
        new_target = deepcopy(target)
        assert op_arg is not None and op_arg != new_target.bg_color
        new_target.target_array[new_target.target_array == new_target.fg_color] = op_arg
        new_target.fg_color = op_arg
        new_target.reset_attributes(op_arg, new_target.bg_color, target.symmetric_type, target.repetitive_type)
        return new_target

    def translation(self, target:SingleColoredTarget, op_arg:Tuple[int,int]=None):
        # translate target by (dy, dx) pixels; out-of-bound pixels are clipped to bbox border
        # no reference to other targets
        dy, dx = op_arg
        new_target = deepcopy(target)
        new_array = np.full_like(new_target.target_array, new_target.bg_color)
        h, w = new_target.target_array.shape
        # compute source and destination slices
        src_row_start = max(0, -dy)
        src_row_end   = min(h, h - dy)
        src_col_start = max(0, -dx)
        src_col_end   = min(w, w - dx)
        dst_row_start = max(0, dy)
        dst_row_end   = dst_row_start + (src_row_end - src_row_start)
        dst_col_start = max(0, dx)
        dst_col_end   = dst_col_start + (src_col_end - src_col_start)
        new_array[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
            target.target_array[src_row_start:src_row_end, src_col_start:src_col_end]
        new_target.target_array = new_array
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target

    def concatenate(self, target1:SingleColoredTarget, target2:SingleColoredTarget, op_arg:str=None):
        # concatenate two targets side by side; op_arg is "horizontal" or "vertical"
        # the resulting array uses both targets' original colors
        assert op_arg in ("horizontal", "vertical")
        new_target = deepcopy(target1)
        if op_arg == "horizontal":
            # align heights by padding the shorter one with bg_color at the bottom
            h = max(target1.bbox_height, target2.bbox_height)
            arr1 = np.full((h, target1.bbox_width), target1.bg_color, dtype=target1.target_array.dtype)
            arr2 = np.full((h, target2.bbox_width), target1.bg_color, dtype=target2.target_array.dtype)
            arr1[:target1.bbox_height, :] = target1.target_array
            arr2[:target2.bbox_height, :] = target2.target_array
            new_array = np.concatenate([arr1, arr2], axis=1)
        else: # vertical
            # align widths by padding the narrower one with bg_color on the right
            w = max(target1.bbox_width, target2.bbox_width)
            arr1 = np.full((target1.bbox_height, w), target1.bg_color, dtype=target1.target_array.dtype)
            arr2 = np.full((target2.bbox_height, w), target1.bg_color, dtype=target2.target_array.dtype)
            arr1[:, :target1.bbox_width] = target1.target_array
            arr2[:, :target2.bbox_width] = target2.target_array
            new_array = np.concatenate([arr1, arr2], axis=0)
        new_target.target_array = new_array
        new_target.bbox_height, new_target.bbox_width = new_array.shape
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target

    def connect(self, image_sampler:ImageSampler, op_arg=None):
        # connect 2 points in the image with a line of the given color
        # op_arg is ((row1,col1), (row2,col2), color)
        (row1, col1), (row2, col2), color = op_arg
        new_image = deepcopy(image_sampler)
        # draw line using Bresenham's algorithm
        dr = abs(row2 - row1)
        dc = abs(col2 - col1)
        sr = 1 if row1 < row2 else -1
        sc = 1 if col1 < col2 else -1
        err = dr - dc
        r, c = row1, col1
        while True:
            if 0 <= r < new_image.image_height and 0 <= c < new_image.image_width:
                new_image.image_array[r, c] = color
            if r == row2 and c == col2:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return new_image

    def shoot(self, image_sampler:ImageSampler, op_arg=None):
        # shoot a ray from a point in 4 directions until the border, filling with the given color
        # op_arg is ((row, col), direction, color); direction is "up"/"down"/"left"/"right"
        (row, col), direction, color = op_arg
        assert direction in ("up", "down", "left", "right")
        new_image = deepcopy(image_sampler)
        r, c = row, col
        while 0 <= r < new_image.image_height and 0 <= c < new_image.image_width:
            new_image.image_array[r, c] = color
            if direction == "up":    r -= 1
            elif direction == "down":  r += 1
            elif direction == "left":  c -= 1
            else:                      c += 1
        return new_image
    
    def hupscale(self, target:SingleColoredTarget, op_arg:int=None):
        # horizontally upscale the target by a factor of op_arg (integer > 1)
        assert op_arg is not None and int(op_arg) > 1
        op_arg = int(op_arg)
        new_target = deepcopy(target)
        h, w = new_target.target_array.shape
        new_w = w * op_arg
        new_array = np.full((h, new_w), new_target.bg_color, dtype=new_target.target_array.dtype)
        for i in range(w):
            new_array[:, i*op_arg:(i+1)*op_arg] = np.expand_dims(new_target.target_array[:, i], axis=1)
        new_target.target_array = new_array
        new_target.bbox_width = new_w
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target

    def vupscale(self, target:SingleColoredTarget, op_arg:int=None):
        # vertically upscale the target by a factor of op_arg (integer > 1)
        assert op_arg is not None and int(op_arg) > 1
        op_arg = int(op_arg)
        new_target = deepcopy(target)
        h, w = new_target.target_array.shape
        new_h = h * op_arg
        new_array = np.full((new_h, w), new_target.bg_color, dtype=new_target.target_array.dtype)
        for i in range(h):
            new_array[i*op_arg:(i+1)*op_arg, :] = np.expand_dims(new_target.target_array[i, :], axis=0)
        new_target.target_array = new_array
        new_target.bbox_height = new_h
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target

    def hdownscale(self, target:SingleColoredTarget, op_arg:int=None):
        # horizontally downscale the target by a factor of op_arg (integer > 1)
        # each output column takes the fg_color if any pixel in the block is fg, else bg_color
        assert op_arg is not None and int(op_arg) > 1
        op_arg = int(op_arg)
        new_target = deepcopy(target)
        h, w = new_target.target_array.shape
        new_w = w // op_arg
        new_array = np.full((h, new_w), new_target.bg_color, dtype=new_target.target_array.dtype)
        for i in range(new_w):
            block = new_target.target_array[:, i*op_arg:(i+1)*op_arg]  # shape (h, op_arg)
            fg_present = np.any(block == new_target.fg_color, axis=1)  # shape (h,)
            new_array[:, i] = np.where(fg_present, new_target.fg_color, new_target.bg_color)
        new_target.target_array = new_array
        new_target.bbox_width = new_w
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target

    def vdownscale(self, target:SingleColoredTarget, op_arg:int=None):
        # vertically downscale the target by a factor of op_arg (integer > 1)
        # each output row takes the fg_color if any pixel in the block is fg, else bg_color
        assert op_arg is not None and int(op_arg) > 1
        op_arg = int(op_arg)
        new_target = deepcopy(target)
        h, w = new_target.target_array.shape
        new_h = h // op_arg
        new_array = np.full((new_h, w), new_target.bg_color, dtype=new_target.target_array.dtype)
        for i in range(new_h):
            block = new_target.target_array[i*op_arg:(i+1)*op_arg, :]  # shape (op_arg, w)
            fg_present = np.any(block == new_target.fg_color, axis=0)  # shape (w,)
            new_array[i, :] = np.where(fg_present, new_target.fg_color, new_target.bg_color)
        new_target.target_array = new_array
        new_target.bbox_height = new_h
        new_target.reset_attributes(new_target.fg_color, new_target.bg_color, "none", "none")
        return new_target


    #####
    
    def get_func(self, assigned_op_name=None, assigned_op_arg=None):
        """
        Sample a func if assigned_op_name is None.
        Otherwise return assigned_op_name and assigned_op_arg
        """
        # get function
        if assigned_op_name is None and assigned_op_arg is None:
            func = random.choice(self.func_list)
            op_arg = "none"
        else:
            func = assigned_op_name
            op_arg = assigned_op_arg
        
        # get arg
        if op_arg == "none":
            if func == "rotation":
                op_arg = random.choice(self.all_rot_degrees)
            elif func == "flip":
                op_arg = random.choice(self.all_flip_axis)
            elif func == "color":
                op_arg = random.choice(self.all_colors)
            elif func == "translation":
                op_arg = random.choice(self.all_translation_offsets)
            elif func == "concatenate":
                op_arg = random.choice(self.all_concat_directions)
            elif func in ("hupscale", "vupscale", "hdownscale", "vdownscale"):
                op_arg = random.choice(self.all_scale_factors)
            # connect and shoot: op_arg must be provided externally (contains coords/direction/color)
        
        return (func, op_arg)
    
    def apply_func(self, target:SingleColoredTarget, assigned_op_name, assigned_op_arg,
                   target2:SingleColoredTarget=None, image_sampler:ImageSampler=None):
        func_call = getattr(self, assigned_op_name)
        if assigned_op_name == "concatenate":
            assert target2 is not None, "concatenate requires target2"
            return func_call(target, target2, assigned_op_arg)
        elif assigned_op_name in ("connect", "shoot"):
            assert image_sampler is not None, f"{assigned_op_name} requires image_sampler"
            return func_call(image_sampler, assigned_op_arg)
        else:
            return func_call(target, assigned_op_arg)