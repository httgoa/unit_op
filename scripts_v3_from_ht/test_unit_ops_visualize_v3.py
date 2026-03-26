import argparse
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from target_sampler_v3 import SingleColoredTarget
from image_sampler_v3 import ImageSampler
from unit_op_und_funcs_v3 import UnitOpUndFuncSampler
from unit_op_gen_funcs_v3 import UnitOpGenFuncSampler


class UnitOpsVisualizerV3:
    def __init__(self, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        self.und = UnitOpUndFuncSampler()
        self.gen = UnitOpGenFuncSampler()

    def _sample_target(self, bbox_min=12, bbox_max=18, bg_color=0):
        success = False
        target = None
        while not success:
            symmetric = random.choice([True, False])
            repetitive = random.choice([True, False])
            if symmetric and repetitive:
                symmetric = False
                repetitive = False
            target = SingleColoredTarget(
                bbox_min,
                bbox_max,
                bg_color=bg_color,
                filled=random.choice([True, False]),
                symmetric=symmetric,
                repetitive=repetitive,
            )
            fg_pixels = np.sum(target.target_array != target.bg_color)
            success = fg_pixels > 0
        return target

    def _sample_color_not_bg(self, bg_color):
        candidates = [c for c in range(0, 10) if c != bg_color]
        return random.choice(candidates)

    def _save_side_by_side(self, before_arr, after_arr, title, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        cmap = plt.cm.get_cmap("tab20", 11)

        axes[0].imshow(before_arr, cmap=cmap, vmin=0, vmax=10, interpolation="nearest")
        axes[0].set_title("Before")
        axes[0].axis("off")

        axes[1].imshow(after_arr, cmap=cmap, vmin=0, vmax=10, interpolation="nearest")
        axes[1].set_title("After")
        axes[1].axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        fig.savefig(save_path, dpi=140)
        plt.close(fig)

    def _run_target_op_case(self, op_name, op_arg=None, bbox_min=12, bbox_max=18):
        target = self._sample_target(bbox_min=bbox_min, bbox_max=bbox_max, bg_color=random.choice(range(0, 10)))
        before = deepcopy(target.target_array)

        if op_name == "color" and (op_arg is None or op_arg == "none"):
            op_arg = self._sample_color_not_bg(target.bg_color)
        elif op_name in ("translation",) and (op_arg is None or op_arg == "none"):
            op_arg = (2, -2)
        elif op_name in ("hupscale", "vupscale", "hdownscale", "vdownscale") and (op_arg is None or op_arg == "none"):
            op_arg = 2
        elif op_arg is None:
            op_arg = "none"

        if hasattr(self.und, op_name):
            op_func = getattr(self.und, op_name)
            after_target = op_func(target, op_arg)
        elif hasattr(self.gen, op_name):
            op_func = getattr(self.gen, op_name)
            after_target = op_func(target, op_arg)
        else:
            raise ValueError(f"Unsupported target op: {op_name}")

        after = after_target.target_array
        return before, after, op_arg

    def _run_concatenate_case(self, op_arg="horizontal"):
        bg_color = random.choice(range(0, 10))
        t1 = self._sample_target(10, 16, bg_color=bg_color)
        t2 = self._sample_target(8, 14, bg_color=bg_color)

        # before: put two targets into a single canvas for visualization
        canvas_h = max(t1.bbox_height, t2.bbox_height)
        canvas_w = t1.bbox_width + t2.bbox_width + 2
        before = np.full((canvas_h, canvas_w), bg_color, dtype=np.int64)
        before[: t1.bbox_height, : t1.bbox_width] = t1.target_array
        before[: t2.bbox_height, t1.bbox_width + 2 : t1.bbox_width + 2 + t2.bbox_width] = t2.target_array

        after_target = self.gen.concatenate(t1, t2, op_arg=op_arg)
        after = after_target.target_array
        return before, after, op_arg

    def _run_image_op_case(self, op_name, op_arg):
        bg_color = random.choice(range(0, 10))
        image = ImageSampler((32, 32), bg_color)
        before = deepcopy(image.image_array)

        op_func = getattr(self.gen, op_name)
        image_after = op_func(image, op_arg)
        after = image_after.image_array
        return before, after, op_arg

    def run(self, output_dir, max_cases=None):
        os.makedirs(output_dir, exist_ok=True)

        cases = [
            ("check_symmetry", "none", "target"),
            ("check_repetition", "none", "target"),
            ("check_holes", "none", "target"),
            ("check_shape", "square", "target"),
            ("leastcolor_target", "none", "target"),
            ("mostcolor_target", "none", "target"),
            ("get_uppermost_target", "none", "target"),
            ("get_lowermost_target", "none", "target"),
            ("get_leftmost_target", "none", "target"),
            ("get_rightmost_target", "none", "target"),
            ("get_background", "none", "target"),
            ("get_size", "none", "target"),
            ("get_width", "none", "target"),
            ("get_height", "none", "target"),
            ("rotation", 1, "target"),
            ("flip", 0, "target"),
            ("delete", "none", "target"),
            ("fill", "none", "target"),
            ("color", "none", "target"),
            ("translation", (2, -2), "target"),
            ("hupscale", 2, "target"),
            ("vupscale", 2, "target"),
            ("hdownscale", 2, "target"),
            ("vdownscale", 2, "target"),
            ("concatenate", "horizontal", "concat"),
            ("concatenate", "vertical", "concat"),
            ("connect", ((3, 3), (28, 25), 2), "image"),
            ("shoot", ((16, 16), "right", 3), "image"),
        ]

        if max_cases is not None:
            cases = cases[:max_cases]

        for idx, (op_name, op_arg, mode) in enumerate(cases):
            try:
                if mode == "target":
                    before, after, used_arg = self._run_target_op_case(op_name, op_arg)
                elif mode == "concat":
                    before, after, used_arg = self._run_concatenate_case(op_arg=op_arg)
                elif mode == "image":
                    before, after, used_arg = self._run_image_op_case(op_name, op_arg)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                title = f"{op_name} | arg={used_arg}"
                save_name = f"{idx:02d}_{op_name}.png"
                self._save_side_by_side(before, after, title, os.path.join(output_dir, save_name))
                print(f"[OK] {save_name}")
            except Exception as err:
                print(f"[FAIL] {op_name} arg={op_arg} -> {err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./unit_ops_cases")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=None)
    args = parser.parse_args()

    runner = UnitOpsVisualizerV3(seed=args.seed)
    runner.run(output_dir=args.output_dir, max_cases=args.max_cases)


if __name__ == "__main__":
    main()
