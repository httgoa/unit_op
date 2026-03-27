import random
import numpy as np
import os
import math
from scipy.ndimage import label
from typing import List, Tuple, Dict
from random_rect_sampler import sample_target

class SingleColoredTarget():
    """
    A single colored target
    """
    def __init__(
        self,
        bbox_size_min,
        bbox_size_max,
        fg_color=None,
        bg_color=None,
        filled=None,
        symmetric=None,
        repetitive=None,
    ):
        # sample color
        color_list = list(range(0, 10)) # 10 reserved for labeling
        
        if bg_color is None:
            if fg_color is not None:
                bg_color = random.choice([c for c in color_list if c!=fg_color])
            else:
                bg_color = random.choice(color_list)
        
        if fg_color is None:
            fg_color = random.choice([c for c in color_list if c!=bg_color])
        
        # get target
        self.target_array, symmetric_type, repetitive_type = sample_target(
            bbox_size_min, 
            bbox_size_max,
            fg_color=fg_color,
            bg_color=bg_color,
            filled=filled,
            symmetric=symmetric,
            repetitive=repetitive
        )
        
        # Reset all attributes based on the newly generated target_array
        self.reset_attributes(fg_color, bg_color, symmetric_type, repetitive_type)
    
    def reset_attributes(self, fg_color, bg_color, symmetric_type, repetitive_type):
        # pixel_list: list of indices of all foreground pixels
        self.pixel_list = self._get_pixel_list(self.target_array, fg_color)
        # fg_size: number of foreground pixels
        self.fg_size = len(self.pixel_list)
        # bounding box size
        self.bbox_height = self.target_array.shape[0]
        self.bbox_width = self.target_array.shape[1]
        # colors
        self.fg_color = fg_color
        self.bg_color = bg_color
        
        # symmetric
        self.symmetric = True if symmetric_type!="none" and not None else False
        self.symmetric_type = symmetric_type
        # repetitive
        self.repetitive = True if repetitive_type!="none" and not None else False
        self.repetitive_type = repetitive_type
        
        # topmost pixels, bottommost pixels, leftmost pixels, rightmost pixels
        self.um_list = self._get_most_list(self.target_array, self.pixel_list, most_type="top")
        self.bm_list = self._get_most_list(self.target_array, self.pixel_list, most_type="bottom")
        self.lm_list = self._get_most_list(self.target_array, self.pixel_list, most_type="left")
        self.rm_list = self._get_most_list(self.target_array, self.pixel_list, most_type="right")

        # outermost pixels
        self.om_list = self._get_frontier_list(self.target_array, self.pixel_list)

        # check if the target has holes
        self.hole_list = self._get_hole_list(self.target_array, self.bg_color)
        self.num_holes = len(self.hole_list)
        self.filled = self.num_holes==0

        # bg_size: number of background pixels
        self.bg_size = self.bbox_height * self.bbox_width - self.fg_size
        # # number of holes
        # self.num_holes = 0 if self.hole_list is None else len(self.hole_list)
        
        
    def _get_pixel_list(self, arr, fg_color):
        h, w = arr.shape
        pixels = []
        rows, cols = np.where(arr==fg_color)
        pixels = list(zip(rows.tolist(), cols.tolist()))
        # sort the pixels first from top to bottom, then from left to right
        pixels = sorted(pixels, key=lambda coord:coord[0]*w+coord[1])
        return pixels
    
    def _get_most_list(self, arr, pixel_list, most_type):
        h, w = arr.shape
        if most_type == 'top':
            result = [(row, col) for (row, col) in pixel_list if row==0]
        elif most_type == 'bottom':
            result = [(row, col) for (row, col) in pixel_list if row==h-1]
        elif most_type == 'left':
            result = [(row, col) for (row, col) in pixel_list if col==0]
        elif most_type == 'right':
            result = [(row, col) for (row, col) in pixel_list if col==w-1]
        return result
    
    def _get_frontier_list(self, arr, pixel_list):
        h, w = arr.shape
        tops, bottoms, lefts, rights = dict(), dict(), dict(), dict()
        for row, col in pixel_list:
            # iteration starts from top to bottom rows, from left to right columns
            # only update topmost when col first appeared during iteration
            if col not in tops:
                tops[col] = row
            # always update bottommost in current column
            bottoms[col] = row
            # only update leftmost when row first appeared during iteration
            if row not in lefts:
                lefts[row] = col
            # always update rightmost in current row
            rights[row] = col
        result = [(row, col) for (col, row) in tops.items()]
        result += [(row, col) for (col, row) in bottoms.items()]
        result += [(row, col) for (row, col) in lefts.items()]
        result += [(row, col) for (row, col) in rights.items()]
        result = list(set(result))
        result = sorted(result, key=lambda coord:coord[0]*w+coord[1])
        return result
    
    
    def _get_hole_list(self, arr: np.ndarray, bg_color: int) -> List[List[Tuple[int, int]]]:
        """
        Finds disjoint holes (background pixels) completely enclosed within an object.
        
        Args:
            arr (np.ndarray): A 2D NumPy array of integers.
            bg_color (int): The integer value representing the background.
            
        Returns:
            List[List[Tuple[int, int]]]: A list where each element is a list of (row, col) 
                                        coordinates forming a single enclosed hole.
        """
        # 1. Create a binary mask of just the background pixels
        bg_mask = (arr == bg_color)
        
        # 2. Label the connected components of the background pixels. 
        # (By default, 'label' uses 4-connectivity, which is standard for hole-finding).
        labeled_bg, num_features = label(bg_mask)
        
        # 3. Identify which background components touch the border of the array.
        # Components touching the border are the "outside" background, not internal holes.
        border_labels = set()
        border_labels.update(labeled_bg[0, :])   # Top row
        border_labels.update(labeled_bg[-1, :])  # Bottom row
        border_labels.update(labeled_bg[:, 0])   # Left column
        border_labels.update(labeled_bg[:, -1])  # Right column

        holes = []
        # 4. Iterate through all found background components
        for i in range(1, num_features + 1):
            # If the component doesn't touch the border, it is trapped inside the object (a hole)
            if i not in border_labels:
                rows, cols = np.where(labeled_bg == i)
                hole_coords = list(zip(rows.tolist(), cols.tolist()))
                holes.append(hole_coords)
        return holes