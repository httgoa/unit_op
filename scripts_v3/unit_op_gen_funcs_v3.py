import os
import numpy as np
import random
from copy import deepcopy
from target_sampler_v3 import SingleColoredTarget

######### unit op functions that perform some actions #########
#平移，翻转，旋转，缩放，取最小单元，重复
# TODO: after each operation, you should update relevant target properties!!!!

class UnitOpGenFuncSampler():
    def __init__(self):
        self.func_list = [
            #"translation",#needs BBox
            "rotation",
            "flip",
            #"scale",#needs BBox
            #"unit", #too hard
            #"repeat",#needs BBox
            "delete",
            #"color",
            "fill",
        ]
        
        self.all_rot_degrees = [1,2,3]
        self.all_flip_axis = [0,1] 
        
    
    def rotation(self, target:SingleColoredTarget, op_arg:str=None):
        assert int(op_arg) in [1, 2, 3]
        rot90 = int(op_arg)
        new_target = deepcopy(target)
        new_array = np.rot90(new_target.target_array, k=rot90)
        new_target.target_array = new_array
        # TODO: update relevant target attributes
        return new_target
        
    def flip(self, target:SingleColoredTarget, op_arg:str=None):
        assert int(op_arg) in [0, 1]
        axis = int(op_arg)
        new_target = deepcopy(target)
        new_array = np.flip(new_target.target_array, axis=axis)
        new_target.target_array = new_array
        # TODO: update relevant target attributes
        return new_target
        
    def delete(self, target:SingleColoredTarget, op_arg:str=None):
        new_target = deepcopy(target)
        new_array = new_target.target_array
        new_array[new_array!=new_target.bg_color] = new_target.bg_color
        # TODO: update relevant target attributes
        return new_target
    
    # should also implement filling some part of the holes of the target
    def fill(self, target:SingleColoredTarget, op_arg:str=None):
        new_target = deepcopy(target)
        if target.hole_list:
            for hole in target.hole_list:
                for point in hole:
                    new_target.target_array[point] = new_target.fg_color
        # TODO: update relevant target attributes
        return new_target
    
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
        
        return (func, op_arg)
    
    def apply_func(self, target:SingleColoredTarget, assigned_op_name, assigned_op_arg):
        func_call = getattr(self, assigned_op_name)
        new_target = func_call(target, assigned_op_arg)
        return new_target