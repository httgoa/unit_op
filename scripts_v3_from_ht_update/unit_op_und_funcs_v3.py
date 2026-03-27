import os
import numpy
import random
from target_sampler_v3 import SingleColoredTarget
from image_sampler_v3 import ImageSampler  
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
            "check_shape",
            "leastcolor_target",
            "mostcolor_target",
            "get_uppermost_target",
            "get_lowermost_target",
            "get_leftmost_target",
            "get_rightmost_target",
            "get_background",
            "get_size",
            "get_width",
            "get_height",
        ]
        self.all_symmetric_types = ["x_axis","y_axis","none"]
        self.all_repetitive_types = ["x_axis","y_axis","none"]
        self.all_shape_types = ["square", "wider", "taller"] # h=w / h<w / h>w
        self.all_colors = list(range(0, 10)) # 0-9, 10 reserved for labeling
        # functions that always produce a changed (labeled) target regardless of target content
        self.always_changed_funcs = [
            "leastcolor_target", "mostcolor_target",
            "get_uppermost_target", "get_lowermost_target",
            "get_leftmost_target", "get_rightmost_target",
            "get_background", "get_size", "get_width", "get_height",
        ]
    
    def check_repetition(self, target:SingleColoredTarget, op_arg:str=None):
        # check if target is repetitive; return labeled target if True, else original
        if op_arg is None or op_arg == "none":
            matches = target.repetitive
        else:
            assert op_arg in self.all_repetitive_types
            matches = (target.repetitive_type == op_arg)
        return self.label(target, need_label=matches)
    
    def check_symmetry(self, target:SingleColoredTarget, op_arg: str=None):
        # check if target is symmetric; return labeled target if True, else original
        if op_arg is None or op_arg == "none":
            matches = target.symmetric
        else:
            assert op_arg in self.all_symmetric_types
            matches = (target.symmetric_type == op_arg)
        return self.label(target, need_label=matches)
    
    def check_holes(self, target:SingleColoredTarget, op_arg:str=None):
        # check if target has holes; return labeled target if True, else original
        matches = (target.num_holes > 0)
        return self.label(target, need_label=matches)
    
    ###### other und ops from ht
    def check_shape(self, target:SingleColoredTarget, op_arg:str=None):
        # check if the target is the shape type (square: h=w / wider: h<w / taller: h>w)
        # return labeled target (bg labeled) if condition holds, else return unlabeled target
        if op_arg is None or op_arg == "none":
            op_arg = random.choice(self.all_shape_types)
        assert op_arg in self.all_shape_types
        h, w = target.bbox_height, target.bbox_width
        if op_arg == "square":
            matches = (h == w)
        elif op_arg == "wider":
            matches = (w > h)
        else: # taller
            matches = (h > w)
        return self.label(target, need_label=matches)

    def get_all_targets(self, image_sampler:ImageSampler, op_arg=None): #copy all the targets and label them
        targets = image_sampler.get_all_targets()
        labeled_targets = []
        for target in targets:
            labeled_target = self.label(target, need_label=True)
            labeled_targets.append(labeled_target)
        return labeled_targets
    
    def leastcolor_target(self, target:SingleColoredTarget, op_arg=None): # get the least frequent color in the target and label corresponding pixels
        # count frequency of each color in the target (excluding bg)
        color_counts = {}
        for (row, col) in target.pixel_list:
            c = target.target_array[row, col]
            color_counts[c] = color_counts.get(c, 0) + 1
        least_color = min(color_counts, key=lambda c: color_counts[c])
        least_pixels = [(row, col) for (row, col) in target.pixel_list
                        if target.target_array[row, col] == least_color]
        return self.label_pixels(target, least_pixels)

    def leastcolor_image(self, image_sampler:ImageSampler, op_arg=None): # for each target in image, label pixels of least frequent color
        targets = image_sampler.get_all_targets()
        labeled_targets = []
        for target in targets:
            labeled_target = self.leastcolor_target(target)
            labeled_targets.append(labeled_target)
        return labeled_targets

    def mostcolor_target(self, target:SingleColoredTarget, op_arg=None): # get the most frequent color in the target and label corresponding pixels
        # count frequency of each color in the target (excluding bg)
        color_counts = {}
        for (row, col) in target.pixel_list:
            c = target.target_array[row, col]
            color_counts[c] = color_counts.get(c, 0) + 1
        most_color = max(color_counts, key=lambda c: color_counts[c])
        most_pixels = [(row, col) for (row, col) in target.pixel_list
                       if target.target_array[row, col] == most_color]
        return self.label_pixels(target, most_pixels)

    def mostcolor_image(self, image_sampler:ImageSampler, op_arg=None): # for each target in image, label pixels of most frequent color
        targets = image_sampler.get_all_targets()
        labeled_targets = []
        for target in targets:
            labeled_target = self.mostcolor_target(target)
            labeled_targets.append(labeled_target)
        return labeled_targets

    def get_uppermost_target(self, target:SingleColoredTarget, op_arg=None): # label the uppermost (topmost row) pixels of the target
        return self.label_pixels(target, target.um_list)

    def get_lowermost_target(self, target:SingleColoredTarget, op_arg=None): # label the lowermost (bottommost row) pixels of the target
        return self.label_pixels(target, target.bm_list)

    def get_leftmost_target(self, target:SingleColoredTarget, op_arg=None): # label the leftmost (leftmost col) pixels of the target
        return self.label_pixels(target, target.lm_list)

    def get_rightmost_target(self, target:SingleColoredTarget, op_arg=None): # label the rightmost (rightmost col) pixels of the target
        return self.label_pixels(target, target.rm_list)

    def get_uppermost_image(self, image_sampler:ImageSampler, op_arg=None): # find the target that sits highest in the image and label it
        targets = image_sampler.get_all_targets()
        selected = min(targets, key=lambda t: min(row for (row, col) in t.pixel_list))
        labeled_targets = []
        for target in targets:
            need_label = (target is selected)
            labeled_targets.append(self.label(target, need_label))
        return labeled_targets

    def get_lowermost_image(self, image_sampler:ImageSampler, op_arg=None): # find the target that sits lowest in the image and label it
        targets = image_sampler.get_all_targets()
        selected = max(targets, key=lambda t: max(row for (row, col) in t.pixel_list))
        labeled_targets = []
        for target in targets:
            need_label = (target is selected)
            labeled_targets.append(self.label(target, need_label))
        return labeled_targets

    def get_leftmost_image(self, image_sampler:ImageSampler, op_arg=None): # find the target that sits furthest left in the image and label it
        targets = image_sampler.get_all_targets()
        selected = min(targets, key=lambda t: min(col for (row, col) in t.pixel_list))
        labeled_targets = []
        for target in targets:
            need_label = (target is selected)
            labeled_targets.append(self.label(target, need_label))
        return labeled_targets

    def get_rightmost_image(self, image_sampler:ImageSampler, op_arg=None): # find the target that sits furthest right in the image and label it
        targets = image_sampler.get_all_targets()
        selected = max(targets, key=lambda t: max(col for (row, col) in t.pixel_list))
        labeled_targets = []
        for target in targets:
            need_label = (target is selected)
            labeled_targets.append(self.label(target, need_label))
        return labeled_targets

    def get_color_target(self, target:SingleColoredTarget, op_arg:int): # label the pixels of the target that match the specified color
        color_pixels = [(row, col) for (row, col) in target.pixel_list
                        if target.target_array[row, col] == op_arg]
        return self.label_pixels(target, color_pixels)
    
    def get_color_image(self, image_sampler:ImageSampler, op_arg:int): # for each target in the image, label the pixels that match the specified color
        targets = image_sampler.get_all_targets()
        labeled_targets = []
        for target in targets:
            labeled_target = self.get_color_target(target, op_arg)
            labeled_targets.append(labeled_target)
        return labeled_targets

    def get_background(self,target:SingleColoredTarget, op_arg=None): # label the background pixels of the target bbox
        bg_pixels = [(row, col) for row in range(target.bbox_height) for col in range(target.bbox_width)
                     if target.target_array[row, col] == target.bg_color]
        return self.label_pixels(target, bg_pixels)
    
    def get_size(self, target:SingleColoredTarget, op_arg=None): 
        # count the number of fg pixels of one target and fill a 5*X rectangle in the top left corner of the image with that number
        count = target.fg_size
        return self.label_count(target, count)

    def get_width(self, target:SingleColoredTarget, op_arg=None):
        # get the width of the target bbox and fill a 5*X rectangle in the top left corner of the image with that number
        count = target.bbox_width
        return self.label_count(target, count)

    def get_height(self, target:SingleColoredTarget, op_arg=None):
        # get the height of the target bbox and fill a 5*X rectangle in the top left corner of the image with that number
        count = target.bbox_height
        return self.label_count(target, count)

    def check_shape(self, target:SingleColoredTarget, op_arg:str=None):
        # check if the target is the shape type (square: h=w / wider: h<w / taller: h>w)
        # return labeled target (bg labeled) if condition holds, else return unlabeled target
        if op_arg is None or op_arg == "none":
            op_arg = random.choice(self.all_shape_types)
        assert op_arg in self.all_shape_types
        h, w = target.bbox_height, target.bbox_width
        if op_arg == "square":
            matches = (h == w)
        elif op_arg == "wider":
            matches = (w > h)
        else: # taller
            matches = (h > w)
        return self.label(target, need_label=matches)
    
    ######

    
    def label_count(self, target:SingleColoredTarget, count:int): #暂未整合进image中
        # return a new Target object where the first `count` pixels (row-major, 5 per row)
        # in the top-left corner are filled with fg_color to encode the numeric value
        new_target = deepcopy(target)
        new_target.target_array[:] = target.bg_color
        filled = 0
        row, col = 0, 0
        while filled < count and row < target.bbox_height:
            new_target.target_array[row, col] = target.fg_color
            filled += 1
            col += 1
            if col >= 5: # 5 pixels per row
                col = 0
                row += 1
        return new_target

    def label_pixels(self, target:SingleColoredTarget, pixel_list:list):
        # return a new Target object with only the specified pixels labeled
        # all other pixels (including bg and non-selected fg) are replaced with LABEL_COLOR
        new_target = deepcopy(target)
        labeled_array = numpy.full_like(new_target.target_array, LABEL_COLOR)
        for (row, col) in pixel_list:
            labeled_array[row, col] = target.target_array[row, col]
        new_target.target_array = labeled_array
        return new_target

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
            elif func == "check_repetition":
                op_arg = random.choice(self.all_repetitive_types)
            elif func == "check_shape":
                op_arg = random.choice(self.all_shape_types)
        
        return (func, op_arg)
    
    def apply_func(self, target:SingleColoredTarget, assigned_op_name, assigned_op_arg):
        func_call = getattr(self, assigned_op_name)
        new_target = func_call(target, assigned_op_arg)
        return new_target