"""Compose a 15s 2K AI manju short from existing generated video clips.

Simple story line:
  1. final_2k_15s_smooth  (romantic/smooth opening, 2K)
  2. shareA_2k            (character A, 2K)
  3. dance_1080           (dance highlight, 1080 -> 2K)
  4. shareB_2k            (character/scene B, 2K)
  5. anim16               (fast atmosphere/action, 256 -> 2K)
  6. final_2k_15s_smooth tail (closing)

All clips are time-resampled to exactly 15s @ 30fps (450 frames).
Low-res clips are upscaled with FSRCNN/RCAS before encoding.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import av
from PIL import Image

OUT_W = 2048
OUT_H = 2048
FPS = 30
DUR = 15.0

FSRCNN_PATH = r"D:\work\textmodel\fsrcnn_x4_v2.pth"
SRC_FPS = 8.0

CLIPS = [
    (r"D:\work\output\adiff_out\final_2k_15s_smooth.mp4", 0.0, 2.2),
    (r"D:\work\output\adiff_out\shareA_2k.mp4", 0.0, 3.2),
    (r"D:\work\output\adiff_out\dance_1080.mp4", 0.0, 1.2),
    (r"D:\work\output\adiff_out\shareB_2k.mp4", 0.0, 3.6),
    (r"D:\work\output\adiff_out\anim16.mp4", 0.0, 1.8),
    (r"D:\work\output\adiff_out\final_2k_15s_smooth.mp4", 2.6, 2.0),
    (r"D:\work\output\adiff_out\jnitaimei_512.mp4", 0.0, 1.0),
]
# Total nominal duration: 2.2+3.2+1.2+3.6+1.8+2.0+1.0 = 15.0


class FSRCNN(nn.Module):
    def __init__(self, in_ch=3, d=64, s=16, m=4, upscale=4):
        super().__init__()
        self.feats = nn.Sequential()
        self.feats.add_module('0', nn.Conv2d(in_ch, d, 5, padding=2))
        self.feats.add_module('1', nn.PReLU(d))
        self.feats.add_module('2', nn.Conv2d(d, s, 1))
        self.feats.add_module('3', nn.PReLU(s))
        for i in range(m):
            self.feats.add_module(f'{4+i}', nn.Sequential(nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)))
        self.feats.add_module('8', nn.Conv2d(s, d, 3, padding=1))
        self.feats.add_module('9', nn.PReLU(d))
        self.feats.add_module('10', nn.Conv2d(d, in_ch * upscale * upscale, 3, padding=1))
        self.upscale = upscale

    def forward(self, x):
        return F.pixel_shuffle(self.feats(x), self.upscale)


def load_sr(weights_path=FSRCNN_PATH):
    sd = torch.load(weights_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model = FSRCNN(d=sd['feats.0.weight'].shape[0], s=sd['feats.2.weight'].shape[0])
    model.load_state_dict(sd)
    model.eval()
    return model


def rcas(img, sharpness=0.8, limit=0.25 - (1.0 / 16.0)):
    b, c, h, w = img.shape
    pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
    b_ = pad[:, :, :-2, 1:-1]
    d_ = pad[:, :, 2:, 1:-1]
    f_ = pad[:, :, 1:-1, :-2]
    h_ = pad[:, :, 1:-1, 2:]
    e_ = img
    def lum(x): return 0.5 * x[:, 2:3] + (0.5 * x[:, 0:1] + x[:, 1:2])
    bL, dL, eL, fL, hL = lum(b_), lum(d_), lum(e_), lum(f_), lum(h_)
    nz = 0.25 * (bL + dL + fL + hL) - eL
    mx = torch.maximum(torch.maximum(torch.maximum(bL, dL), eL), torch.maximum(fL, hL))
    mn = torch.minimum(torch.minimum(torch.minimum(bL, dL), eL), torch.minimum(fL, hL))
    nz = torch.clamp(torch.abs(nz) / (mx - mn + 1e-6), 0, 1)
    nz = -0.5 * nz + 1.0
    ring_min = torch.minimum(torch.minimum(torch.minimum(b_, d_), f_), h_)
    ring_max = torch.maximum(torch.maximum(torch.maximum(b_, d_), f_), h_)
    hitMin = torch.minimum(ring_min, e_) / (4.0 * ring_max + 1e-8)
    hitMax = (1.0 - torch.maximum(ring_max, e_)) / (4.0 * ring_min - 4.0 + 1e-8)
    lobe_rgb = torch.maximum(-hitMin, hitMax)
    lobe = torch.maximum(torch.tensor(-limit), torch.minimum(lobe_rgb.max(1, keepdim=True).values, torch.tensor(0.0))) * sharpness * nz
    rcpL = 1.0 / (4.0 * lobe + 1.0)
    return torch.clamp((lobe * (b_ + d_ + h_ + f_) + e_) * rcpL, 0, 1)


def read_clip(path, start_s, dur_s):
    """Read frames from a video file for [start_s, start_s+dur_s), yield uint8 numpy frames."""
    c = av.open(path)
    vs = c.streams.video[0]
    fps = float(vs.average_rate)
    start_idx = int(round(start_s * fps))
    end_idx = int(round((start_s + dur_s) * fps))
    frames = []
    for i, f in enumerate(c.decode(video=0)):
        if i < start_idx:
            continue
        if i >= end_idx:
            break
        frames.append(f.to_ndarray(format='rgb24'))
    c.close()
    return frames, fps


def upscale_to_2k(frames, sr_model=None):
    """Convert a list of uint8 RGB frames to OUT_W x OUT_H, using FSRCNN when helpful."""
    out = []
    for fr in frames:
        h, w = fr.shape[:2]
        if w < OUT_W or h < OUT_H:
            # use FSRCNN x4 when input is small and factor is 4-ish
            if sr_model is not None and w * 4 <= OUT_W and h * 4 <= OUT_H and (w % 4 == 0):
                t = torch.from_numpy(fr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    t = sr_model(t)
                up = t[0].permute(1, 2, 0)
                # resize if the 4x output is still smaller than target
                if up.shape[1] != OUT_H or up.shape[2] != OUT_W:
                    up = up.permute(2, 0, 1).unsqueeze(0)
                    up = F.interpolate(up, size=(OUT_H, OUT_W), mode='bicubic', align_corners=False)
                    up = up[0].permute(1, 2, 0)
                arr = (up.clamp(0, 1).numpy() * 255).astype(np.uint8)
            else:
                im = Image.fromarray(fr).resize((OUT_W, OUT_H), Image.BICUBIC)
                arr = np.asarray(im)
            if arr.shape[:2] != (OUT_H, OUT_W):
                im = Image.fromarray(arr).resize((OUT_W, OUT_H), Image.BICUBIC)
                arr = np.asarray(im)
            # RCAS on the final upscaled image
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                t = rcas(t)
            arr = (t[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = fr
            if arr.shape[:2] != (OUT_H, OUT_W):
                im = Image.fromarray(arr).resize((OUT_W, OUT_H), Image.BICUBIC)
                arr = np.asarray(im)
        out.append(arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\work\output\adiff_out\manju_2k_15s.mp4')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    print('[manju] loading FSRCNN', flush=True)
    sr_model = load_sr()

    target_frames = int(DUR * FPS)
    all_frames = []  # timeline entries (each covers nominal duration)

    print('[manju] reading clips', flush=True)
    for path, start, dur in CLIPS:
        frames, fps = read_clip(path, start, dur)
        print(f'  {os.path.basename(path)} start={start}s dur={dur}s -> {len(frames)} frames @{fps:.2f}fps', flush=True)
        frames = upscale_to_2k(frames, sr_model)
        all_frames.append(frames)

    total_src_frames = sum(len(f) for f in all_frames)
    print(f'[manju] total source frames {total_src_frames}', flush=True)

    # Build long source frame list, then time-resample to target_frames.
    # Each clip contributes proportionally to its nominal duration.
    long_frames = []
    nominal_total = sum(dur for _, _, dur in CLIPS)
    for (_, _, dur), frames in zip(CLIPS, all_frames):
        n = max(1, int(round(dur / nominal_total * total_src_frames)))
        # expand/compress this clip to n frames
        if len(frames) == 1:
            long_frames.extend([frames[0]] * n)
        else:
            idx = np.linspace(0, len(frames) - 1, n)
            for pos in idx:
                i0 = int(np.floor(pos))
                i1 = min(i0 + 1, len(frames) - 1)
                alpha = pos - i0
                f = (frames[i0].astype(np.float32) * (1 - alpha) + frames[i1].astype(np.float32) * alpha).astype(np.uint8)
                long_frames.append(f)

    print(f'[manju] long timeline frames {len(long_frames)}, resampling to {target_frames}', flush=True)
    if len(long_frames) > target_frames:
        idx = np.linspace(0, len(long_frames) - 1, target_frames)
    else:
        idx = np.linspace(0, len(long_frames) - 1, target_frames)
    final = []
    for pos in idx:
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, len(long_frames) - 1)
        alpha = pos - i0
        f = (long_frames[i0].astype(np.float32) * (1 - alpha) + long_frames[i1].astype(np.float32) * alpha).astype(np.uint8)
        final.append(f)

    print('[manju] encoding mp4', flush=True)
    out = av.open(a.out, 'w')
    st = out.add_stream('mpeg4', rate=FPS)
    st.width = OUT_W
    st.height = OUT_H
    st.pix_fmt = 'yuv420p'
    st.bit_rate = 20_000_000
    for fr in final:
        frame = av.VideoFrame.from_ndarray(fr, format='rgb24')
        for p in st.encode(frame):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    print(f'[manju] done -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
