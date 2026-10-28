"""Turn generated low-res AnimateDiff frames into a 2K 15s video.

Pipeline:
  input PNGs (e.g. 128x128) -> time-resample to target frame count
  -> FSRCNN x4 to 512 -> FSRCNN x4 to 2048 -> encode mp4

This uses the real trained FSRCNN d64 weights from:
  D:\\work\\textmodel\\fsrcnn_x4_v2.pth
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

FSRCNN_PATH = r"D:\work\textmodel\fsrcnn_x4_v2.pth"


class FSRCNN(nn.Module):
    def __init__(self, in_ch=3, d=64, s=16, m=4, upscale=4):
        super().__init__()
        self.feats = nn.Sequential()
        self.feats.add_module('0', nn.Conv2d(in_ch, d, 5, padding=2))
        self.feats.add_module('1', nn.PReLU(d))
        self.feats.add_module('2', nn.Conv2d(d, s, 1))
        self.feats.add_module('3', nn.PReLU(s))
        for i in range(m):
            self.feats.add_module(f'{4+i}', nn.Sequential(
                nn.Conv2d(s, s, 3, padding=1),
                nn.PReLU(s),
            ))
        self.feats.add_module('8', nn.Conv2d(s, d, 3, padding=1))
        self.feats.add_module('9', nn.PReLU(d))
        self.feats.add_module('10', nn.Conv2d(d, in_ch * upscale * upscale, 3, padding=1))
        self.upscale = upscale

    def forward(self, x):
        x = self.feats(x)
        return F.pixel_shuffle(x, self.upscale)


def load_sr(weights_path=FSRCNN_PATH):
    sd = torch.load(weights_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    d = sd['feats.0.weight'].shape[0]
    s = sd['feats.2.weight'].shape[0]
    model = FSRCNN(d=d, s=s)
    model.load_state_dict(sd)
    model.eval()
    return model


def load_frames(dirpath):
    names = sorted(os.listdir(dirpath))
    names = [n for n in names if n.lower().endswith('.png')]
    frames = []
    for n in names:
        im = Image.open(os.path.join(dirpath, n)).convert('RGB')
        frames.append(np.asarray(im).astype(np.float32) / 255.0)
    if not frames:
        raise RuntimeError(f'no PNG frames in {dirpath}')
    return frames, names


def time_resample(frames, target_count):
    """Linear time-domain resample to exactly target_count frames."""
    n = len(frames)
    if n == 1:
        return [frames[0] for _ in range(target_count)]
    out = []
    # map output index to continuous source position
    # source range [0, n-1]; output positions spread over same interval,
    # but if target_count > n this plays the clip slower (more frames per source)
    positions = np.linspace(0, n - 1, target_count)
    arr = np.stack(frames).astype(np.float32)  # T,H,W,C
    h, w = frames[0].shape[:2]
    arr_t = arr.transpose(0, 3, 1, 2)  # T,C,H,W
    for pos in positions:
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, n - 1)
        alpha = pos - i0
        f = (1 - alpha) * arr_t[i0] + alpha * arr_t[i1]
        out.append(f.transpose(1, 2, 0))
    return out


def sr_upscale(model, frames_nhwc, target_scale=4, threads=6):
    """Upscale each frame by x4 using FSRCNN. Input list of float NHWC."""
    torch.set_num_threads(threads)
    out = []
    with torch.no_grad():
        for fr in frames_nhwc:
            t = torch.from_numpy(fr).permute(2, 0, 1).unsqueeze(0).float()
            y = model(t)
            y = y[0].permute(1, 2, 0).clamp(0, 1).numpy()
            out.append(y)
    return out


def encode_mp4(frames_nhwc, path, fps=8.0, width=None, height=None, crf_style=True):
    import av
    h, w = frames_nhwc[0].shape[:2]
    if width is not None and height is not None:
        h, w = height, width
    out = av.open(path, 'w')
    st = out.add_stream('mpeg4', rate=int(round(fps)))
    st.width = w
    st.height = h
    st.pix_fmt = 'yuv420p'
    st.bit_rate = int(18_000_000)  # high enough for 2K
    for fr in frames_nhwc:
        # fr could be float NHWC 0..1
        if fr.dtype != np.uint8:
            fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
        if fr.shape[:2] != (h, w):
            im = Image.fromarray(fr).resize((w, h), Image.BICUBIC)
            fr = np.asarray(im)
        frame = av.VideoFrame.from_ndarray(fr, format='rgb24')
        for p in st.encode(frame):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=r'D:\work\output\adiff_out\clips_128_8f_3s')
    ap.add_argument('--output', default=r'D:\work\output\adiff_out\result_2k_15s.mp4')
    ap.add_argument('--seconds', type=float, default=15.0)
    ap.add_argument('--fps', type=float, default=8.0)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--sr', default=FSRCNN_PATH)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    target_frames = int(round(a.seconds * a.fps))
    print(f'[sr] load frames from {a.input}', flush=True)
    t0 = time.time()
    frames, names = load_frames(a.input)
    print(f'[sr] loaded {len(frames)} frames {frames[0].shape} in {time.time()-t0:.1f}s', flush=True)

    print(f'[sr] resample {len(frames)} -> {target_frames}', flush=True)
    t0 = time.time()
    frames = time_resample(frames, target_frames)
    print(f'[sr] resample done {time.time()-t0:.1f}s', flush=True)

    print('[sr] load FSRCNN', flush=True)
    t0 = time.time()
    model = load_sr(a.sr)
    print(f'[sr] model loaded {time.time()-t0:.1f}s', flush=True)

    print('[sr] FSRCNN x4 -> 512', flush=True)
    t0 = time.time()
    frames512 = sr_upscale(model, frames, 4, a.threads)
    print(f'[sr] 512 upscale done {time.time()-t0:.1f}s shape={frames512[0].shape}', flush=True)

    print('[sr] FSRCNN x4 -> 2048', flush=True)
    t0 = time.time()
    frames2k = sr_upscale(model, frames512, 4, a.threads)
    print(f'[sr] 2k upscale done {time.time()-t0:.1f}s shape={frames2k[0].shape}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    print('[sr] encode mp4', flush=True)
    t0 = time.time()
    encode_mp4(frames2k, a.output, fps=a.fps)
    print(f'[sr] encode done {time.time()-t0:.1f}s -> {a.output}', flush=True)


if __name__ == '__main__':
    main()
