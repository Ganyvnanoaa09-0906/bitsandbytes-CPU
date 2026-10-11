"""Auxiliary numbers for the step-count curve, not a quality verdict.

The report already established that pixel/relative metrics cannot judge a diffusion
sample: the sampler is chaotic, so two images can differ by 0.3 in relative error and look
identical, or by 0.9 and be a different picture entirely. What these numbers do is
corroborate two specific failure modes that are visible by eye at low step counts:

  * detail collapse  -> Laplacian variance falls (the image goes soft)
  * wash-out         -> the mean rises and the dynamic range narrows (the 4-step sample
                        looked overexposed and flat)

They are printed next to the step count so the visual read and the numbers can be
compared; the verdict stays with the eye.

Usage:
    python qwen21_sharpness.py qwen21_ab/render_euler*.png
"""
import sys

import numpy as np
import torch
from PIL import Image


def laplacian_var(gray: torch.Tensor) -> float:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    x = gray.view(1, 1, *gray.shape)
    lap = torch.nn.functional.conv2d(x, k, padding=1)
    return float(lap.var() * 1e4)          # scaled to a readable range


def main(paths):
    print(f"{'file':<34} {'sharp':>8} {'mean':>7} {'std':>7} {'clip%':>7} "
          f"{'p1':>6} {'p99':>6}")
    for p in paths:
        img = Image.open(p).convert("RGB")
        arr = torch.from_numpy(np.asarray(img)).float() / 255.0
        gray = arr.mean(-1)
        lum = gray
        clip = float(((arr <= 0.02) | (arr >= 0.98)).float().mean() * 100)
        q = torch.quantile(lum.flatten(), torch.tensor([0.01, 0.99]))
        print(f"{p.split(chr(92))[-1].split('/')[-1]:<34} {laplacian_var(gray):>8.1f} "
              f"{lum.mean():>7.3f} {lum.std():>7.3f} {clip:>7.2f} "
              f"{float(q[0]):>6.3f} {float(q[1]):>6.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or [f"qwen21_ab/render_euler{s}.png" for s in (20, 12, 8, 6, 4)])
