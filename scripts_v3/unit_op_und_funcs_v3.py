import os
import numpy
import random
from target_sampler_v3 import SingleColoredTarget
from copy import deepcopy


######### unit op functions that does understands or analysis #########

#是否某种shape；是否对称；是否recursive；是否接触border；是否有holes
#这个function sampler除了针对Target进行query之外，还要针对BBox进行query，暂时先只针对Target

LABEL_COLOR = 10 # 10 reserved for labeling and should not be used in target generations 

class UnitOpUndFuncSampler():
    def __init__(self):
        self.func_list = [
            "check_symmetry",
            "check_holes",
            "check_repetition",
        ]
        self.all_symmetric_types = ["x_axis","y_axis","none"]
        self.all_repetitive_types = ["x_axis","y_axis","none"]
    
    def check_repetition(self, target:SingleColoredTarget, op_arg:str=None):
        if op_arg is None or op_arg == "none":
            return not target.repetitive
        else:
            assert op_arg in self.all_repetitive_types
            return target.repetitive_type == op_arg
    
    def check_symmetry(self, target:SingleColoredTarget, op_arg: str=None):
        if op_arg is None or op_arg == "none":
            return not target.symmetric
        else:
            assert op_arg in self.all_symmetric_types
            return target.symmetric_type == op_arg
    
    def check_holes(self, target:SingleColoredTarget, op_arg:str=None):
        return target.num_holes > 0
    
    def label(self, target:SingleColoredTarget, need_label:bool):
        # return a new Target object after labeling
        # change the background color to labeling color to simplify the training process
        new_target = deepcopy(target)
        if need_label:
            new_target.target_array[new_target.target_array==target.bg_color] = LABEL_COLOR
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
            if func == "check_symmetry":
                op_arg = random.choice(self.all_symmetric_types)
            if func == "check_repetition":
                op_arg = random.choice(self.all_repetitive_types)
        
        return (func, op_arg)
    
    def apply_func(self, target:SingleColoredTarget, assigned_op_name, assigned_op_arg):
        func_call = getattr(self, assigned_op_name)
        need_label = func_call(target, assigned_op_arg)
        # label the resulting image
        new_target = self.label(target, need_label)
        return new_target