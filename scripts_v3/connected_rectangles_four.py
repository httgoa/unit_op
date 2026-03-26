import numpy as np
import random

from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

def detect_holes(array, foreground_color):
    # 找到连通组件内部的背景区域
    background_mask = (array == 0)
    
    # 标记背景区域的连通组件
    labeled_background, num_features = label(background_mask)
    
    # 掩码边界，避免边界上的背景被认为是洞
    for label_idx in range(1, num_features + 1):
        if np.any(labeled_background[0, :] == label_idx) or \
           np.any(labeled_background[-1, :] == label_idx) or \
           np.any(labeled_background[:, 0] == label_idx) or \
           np.any(labeled_background[:, -1] == label_idx):
            labeled_background[labeled_background == label_idx] = 0  # 删除边缘连通区域
    
    # 返回洞掩码和数量
    hole_mask = (labeled_background > 0)
    num_holes = len(np.unique(labeled_background[hole_mask]))  # 洞数量
    return num_holes, hole_mask

def create_holes(array, foreground_color, max_holes=3):
    
    arr_h, arr_w = array.shape
    hole_count = random.randint(1, max_holes)  # 创建 1 到 max_holes 个洞

    for _ in range(hole_count):
        attempt = 0  # 限制尝试次数，避免死循环
        while attempt < 100:  # 最多尝试 100 次
            attempt += 1

            # 随机选择可能的洞位置和尺寸
            h, w = random.randint(3, 10), random.randint(3,10)
            h1, w1 = random.randint(1, arr_h-3),random.randint(1, arr_w-3)  # 保留边界像素，避免洞贴近边界
            h2, w2 = min(arr_h-3, h1+h), min(arr_w-3, w1+w)

            # 提取候选洞的上下左右邻域
            above = array[h1-1, w1:w2]        # 洞上方
            below = array[h2, w1:w2]           # 洞下方
            left = array[h1:h2, w1-1]        # 洞左侧
            right = array[h1:h2, w2]           # 洞右侧

            # 检查洞是否合法：上下左右必须被前景色包围
            if (
                np.all(array[h1:h2, w1:w2] == foreground_color)  # 洞本身属于前景色
                and np.all(above == foreground_color)            # 上方是前景色
                and np.all(below == foreground_color)            # 下方是前景色
                and np.all(left == foreground_color)             # 左边是前景色
                and np.all(right == foreground_color)            # 右边是前景色
            ):
                # 如果合法，则将洞设置为背景色
                array[h1:h2, w1:w2] = 0
                break  # 成功创建一个洞

    return array, hole_count

def get_pixel_list(arr, colors):
    h, w = arr.shape
    pixels = []
    for c in colors:
        rows, cols = np.where(arr == c)
        pixels.extend(list(zip(rows.tolist(), cols.tolist())))
    # sort first from top to bottom, then from left to right
    pixels = sorted(pixels, key=lambda coord:coord[0]*w+coord[1])
    return pixels

def generate_connected_rectangles(arr_height:int=64, arr_width:int=64, fg_color: int = 1) -> np.ndarray:
    array = np.zeros((arr_height, arr_width), dtype=np.int32)

    # the first rectangle
    r1_h, r1_w = random.randint(arr_height//4, arr_height//2), random.randint(arr_width//3, arr_width//2)
    # the first rectangle starts from the top edge
    p1_top, p1_left = 0, random.randint(0, arr_width-r1_w)
    # draw the first rectangle
    array[p1_top:p1_top+r1_h, p1_left:p1_left+r1_w] = fg_color

    # get pixel_list
    current_pixels = get_pixel_list(array, colors=[fg_color])
    # select start point from current pixels
    p2_top, p2_right = random.choice(current_pixels)
    # the second rectangle should touch the left edge
    # [0:p2_right], p2_left=0
    r2_h = random.randint((arr_height-p2_top)//4, (arr_height-p2_top)//2)
    array[p2_top:p2_top+r2_h, 0:p2_right] = fg_color

    # get pixel list
    current_pixels = get_pixel_list(array, colors=[fg_color])
    # select start point from current pixels
    p3_top, p3_left = random.choice(current_pixels)
    # the third rectangle should touch the bottom edge
    # [p3_top:],
    r3_w = random.randint((arr_width-p3_left)//4, (arr_width-p3_left)//2)
    array[p3_top:, p3_left:p3_left+r3_w] = fg_color

    # get pixel list
    current_pixels = get_pixel_list(array, colors=[fg_color])
    # select start point from current pixels
    p4_top, p4_left = random.choice(current_pixels)
    # the fourth rectangle should touch the right edge
    # [p4_left:],
    r4_h = random.randint((arr_height-p4_top)//4, (arr_height-p4_top)//2)
    array[p4_top:p4_top+r4_h, p4_left:] = fg_color

    return array, fg_color


def generate_unit_array(height=64, width=64, filled=True):
    array, foreground_color = generate_connected_rectangles(arr_height=height, arr_width=width)
    num_holes, hole_mask = detect_holes(array, foreground_color)
    
    if filled:
        if num_holes > 0:
            array= binary_fill_holes(array > 0) * foreground_color  # 填洞
            num_holes = 0
    else:
        if num_holes == 0:  # 如果没有洞，则手动创建洞
            array, num_holes = create_holes(array, foreground_color, max_holes=2)
    
    return array, num_holes

def generate_array_binary(bbox_size_min, bbox_size_max, symmetric=False, repetitive=False):
    symmetric_type, repetitive_type = "none", "none"
    if symmetric and repetitive:
        if random.random() < 0.5:
            symmetric_type = "x_axis"
            repetitive_type = "x_axis"
        else:
            symmetric_type = "y_axis"
            repetitive_type = "y_axis"
    elif symmetric and not repetitive:
        symmetric_type = random.choices(["y_axis", "x_axis", "xy_axis"], weights=[1, 1, 3])[0]
    elif not symmetric and repetitive:
        repetitive_type = random.choices(["y_axis","x_axis","xy_axis"], weights=[1,1,3])[0]
        
    unit_height = random.randint(bbox_size_min, bbox_size_max)
    unit_width = random.randint(bbox_size_min, bbox_size_max)
    h_flip_cnt, w_flip_cnt, h_trans_cnt, w_trans_cnt = 1,1,1,1
    
    if symmetric_type in ["x_axis", "xy_axis"]:
        unit_height = random.randint(bbox_size_min//2+1, bbox_size_max//2)
        h_flip_cnt = 2
    if symmetric_type in ["y_axis", "xy_axis"]:
        unit_width = random.randint(bbox_size_min//2+1, bbox_size_max//2)
        w_flip_cnt = 2
    if repetitive_type in ["y_axis", "xy_axis"]:
        h_trans_cnt = random.randint(2, 3)
        unit_height = random.randint(bbox_size_min//h_trans_cnt+1, bbox_size_max//h_trans_cnt)
    if repetitive_type in ["x_axis", "xy_axis"]:
        w_trans_cnt = random.randint(2, 3)
        unit_width = random.randint(bbox_size_min//w_trans_cnt+1, bbox_size_max//w_trans_cnt)
    
    # generate unit arr
    if symmetric or repetitive:
        filled=True
    else:
        filled = random.choice([False, True])
    unit_arr, num_holes = generate_unit_array(height=unit_height, width=unit_width, filled=filled)
    
    # generate complete arr on the basis of unit arr
    # Perform flipping and translation according to flip_cnt and trans_cnt
    complete_arr = unit_arr.copy()
    
    # Apply vertical (x-axis) symmetry
    if h_flip_cnt > 1:
        flipped_arr = np.flipud(unit_arr)  # Flip along the vertical axis
        complete_arr = np.vstack([complete_arr, flipped_arr])
    # Apply horizontal (y-axis) symmetry
    if w_flip_cnt > 1:
        flipped_arr = np.fliplr(complete_arr)  # Flip along the horizontal axis
        complete_arr = np.hstack([complete_arr, flipped_arr])
    # Apply vertical repetitions
    if h_trans_cnt > 1:
        complete_arr = np.tile(complete_arr, (h_trans_cnt, 1))
    # Apply horizontal repetitions
    if w_trans_cnt > 1:
        complete_arr = np.tile(complete_arr, (1, w_trans_cnt))

    return complete_arr





def generate_array_binary_simple(bbox_size_min, bbox_size_max, symmetric=False, repetitive=False):
    symmetric_type, repetitive_type = "none", "none"
    assert not (symmetric and repetitive) # symmetric and repetitive means a combinatory op!!!
    
    if symmetric:
        symmetric_type = random.choice(["y_axis", "x_axis"])
    if repetitive:
        repetitive_type = random.choice(["y_axis","x_axis"])
        
    unit_height = random.randint(bbox_size_min, bbox_size_max)
    unit_width = random.randint(bbox_size_min, bbox_size_max)
    h_flip_cnt, w_flip_cnt, h_trans_cnt, w_trans_cnt = 1,1,1,1
    
    if symmetric_type == "x_axis":
        unit_height = random.randint(bbox_size_min//2+1, bbox_size_max//2)
        h_flip_cnt = 2
    if symmetric_type == "y_axis":
        unit_width = random.randint(bbox_size_min//2+1, bbox_size_max//2)
        w_flip_cnt = 2
    if repetitive_type == "x_axis":
        h_trans_cnt = random.randint(2, 3)
        unit_height = random.randint(bbox_size_min//h_trans_cnt+1, bbox_size_max//h_trans_cnt)
    if repetitive_type == "y_axis":
        w_trans_cnt = random.randint(2, 3)
        unit_width = random.randint(bbox_size_min//w_trans_cnt+1, bbox_size_max//w_trans_cnt)
    
    # generate unit arr
    if symmetric or repetitive:
        filled = True
    else:
        filled = random.choice([False, True])
    unit_arr, num_holes = generate_unit_array(height=unit_height, width=unit_width, filled=filled)
    
    # generate complete arr on the basis of unit arr
    # Perform flipping and translation according to flip_cnt and trans_cnt
    complete_arr = unit_arr.copy()
    
    # Apply vertical (x-axis) symmetry
    if h_flip_cnt > 1:
        flipped_arr = np.flipud(unit_arr)  # Flip along the vertical axis
        complete_arr = np.vstack([complete_arr, flipped_arr])
    # Apply horizontal (y-axis) symmetry
    if w_flip_cnt > 1:
        flipped_arr = np.fliplr(complete_arr)  # Flip along the horizontal axis
        complete_arr = np.hstack([complete_arr, flipped_arr])
    # Apply vertical repetitions
    if h_trans_cnt > 1:
        complete_arr = np.tile(complete_arr, (h_trans_cnt, 1))
    # Apply horizontal repetitions
    if w_trans_cnt > 1:
        complete_arr = np.tile(complete_arr, (1, w_trans_cnt))

    return complete_arr




# Example usage
if __name__ == "__main__":

    # 生成数组（filled=True 或 filled=False）
    size_max = 40
    size_min = 30
    array = generate_array_binary(bbox_size_max=size_max, bbox_size_min=size_min, repetitive=True, symmetric=True)
    h, w= array.shape
    print(f"h={h} and w={w}")
    assert h<=size_max and h>=size_min and w<=size_max and w>=size_min
    # 可视化生成结果
    plt.imshow(array, cmap="tab20")
    plt.colorbar()
    plt.show()
