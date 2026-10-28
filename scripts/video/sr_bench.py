"""Benchmark FSRCNN / RCAS upscaling paths."""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class FSRCNN(nn.Module):
    """FSRCNN with d=64, s=16, m=4, PixelShuffle x4."""
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


def rcas(img, sharpness=0.7, limit=0.25 - (1.0 / 16.0)):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=r'D:\work\textmodel\fsrcnn_x4_v2.pth')
    ap.add_argument('--in_res', type=int, default=512)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    sd = torch.load(a.model, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model = FSRCNN(d=sd['feats.0.weight'].shape[0], s=sd['feats.2.weight'].shape[0])
    model.load_state_dict(sd)
    model.eval()

    x = torch.rand(1, 3, a.in_res, a.in_res)
    with torch.no_grad():
        # warmup
        for _ in range(2):
            y = model(x)
        t0 = time.time()
        y = model(x)
        dt = time.time() - t0
        print(f'FSRCNN {a.in_res}->{a.in_res*a.scale} 1 frame: {dt*1000:.1f} ms, out={tuple(y.shape)}')

        # bicubic + rcas
        t0 = time.time()
        up = F.interpolate(x, scale_factor=a.scale, mode='bicubic', align_corners=False)
        sh = rcas(up)
        dt2 = time.time() - t0
        print(f'bicubic+RCAS {a.in_res}->{a.in_res*a.scale} 1 frame: {dt2*1000:.1f} ms')


if __name__ == '__main__':
    main()
