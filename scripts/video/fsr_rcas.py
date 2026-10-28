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
import torch, time, os, sys
import torch.nn.functional as F
from PIL import Image
import numpy as np

def rcas(img, sharpness=0.25, limit=0.25-(1.0/16.0)):
    """FSR 1.0 RCAS 锐化 (5-tap 十字 + lobe 自适应). img: [B,3,H,W] 0-1"""
    b, c, h, w = img.shape
    # 5 tap: 上b 下d 中心e 左f 右h (用 pad 取邻域)
    pad = F.pad(img, (1,1,1,1), mode='replicate')
    b_ = pad[:,:,:-2, 1:-1]; d_ = pad[:,:,2:, 1:-1]
    f_ = pad[:,:,1:-1, :-2]; h_ = pad[:,:,1:-1, 2:]
    e_ = img
    # luma x2
    def lum(x): return 0.5*x[:,2:3]+(0.5*x[:,0:1]+x[:,1:2])
    bL,dL,eL,fL,hL = lum(b_),lum(d_),lum(e_),lum(f_),lum(h_)
    # noise detect
    nz = 0.25*(bL+dL+fL+hL)-eL
    mx = torch.maximum(torch.maximum(torch.maximum(bL,dL),eL),torch.maximum(fL,hL))
    mn = torch.minimum(torch.minimum(torch.minimum(bL,dL),eL),torch.minimum(fL,hL))
    nz = torch.clamp(torch.abs(nz)/ (mx-mn+1e-6), 0, 1)
    nz = -0.5*nz+1.0
    # min/max ring per channel
    ring_min = torch.minimum(torch.minimum(torch.minimum(b_,d_),f_),h_)
    ring_max = torch.maximum(torch.maximum(torch.maximum(b_,d_),f_),h_)
    peakC0, peakC1 = 1.0, -4.0
    hitMin = torch.minimum(ring_min, e_) / (4.0*ring_max + 1e-8)
    hitMax = (peakC0-torch.maximum(ring_max, e_)) / (4.0*ring_min + peakC1 + 1e-8)
    lobe_rgb = torch.maximum(-hitMin, hitMax)  # [B,3,H,W]
    lobe = torch.maximum(torch.tensor(-limit), torch.minimum(lobe_rgb.max(1,keepdim=True).values, torch.tensor(0.0))) * sharpness * nz
    rcpL = 1.0/(4.0*lobe + 1.0)
    pix = (lobe*(b_+d_+h_+f_) + e_) * rcpL
    return torch.clamp(pix,0,1)

img = Image.open('D:/work/output/adiff_out/sd15_512.png').convert('RGB')
t = torch.from_numpy(np.asarray(img)).permute(2,0,1).unsqueeze(0).float()/255.0
# 上采样到 1920x1080 (bicubic) -> 1080p 大小要 align: 512x512 -> 长边放大; 这里演示 512->512*2=1024, RCAS 锐化
scale = 2
up = F.interpolate(t, scale_factor=scale, mode='bicubic', align_corners=False)
print('上采样后 shape', tuple(up.shape), ' (512x2=%d)'%(512*scale))
t0=time.time()
sharpen = rcas(up, sharpness=0.8)
print('RCAS 锐化 512x2 耗时 %.2fs 内存RSS %.2fGB'%(time.time()-t0, __import__('psutil').Process().memory_info().rss/1e9))
out = (sharpen[0].permute(1,2,0).numpy()*255).astype('uint8')
Image.fromarray(out).save('D:/work/output/adiff_out/fsr_rcas.png')
print('已存 fsr_rcas.png', tuple(out.shape))
