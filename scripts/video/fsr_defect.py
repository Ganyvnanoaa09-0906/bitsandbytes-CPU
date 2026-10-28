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
import torch, numpy as np, psutil
import torch.nn.functional as F
from PIL import Image
def rcas(img, sharpness=0.8, limit=0.25-(1.0/16.0)):
    b,c,h,w=img.shape; pad=F.pad(img,(1,1,1,1),mode='replicate')
    b_=pad[:,:,:-2,1:-1]; d_=pad[:,:,2:,1:-1]; f_=pad[:,:,1:-1,:-2]; h_=pad[:,:,1:-1,2:]; e_=img
    def lum(x): return 0.5*x[:,2:3]+(0.5*x[:,0:1]+x[:,1:2])
    bL,dL,eL,fL,hL=lum(b_),lum(d_),lum(e_),lum(f_),lum(h_)
    nz=0.25*(bL+dL+fL+hL)-eL
    mx=torch.maximum(torch.maximum(torch.maximum(bL,dL),eL),torch.maximum(fL,hL)); mn=torch.minimum(torch.minimum(torch.minimum(bL,dL),eL),torch.minimum(fL,hL))
    nz=torch.clamp(torch.abs(nz)/(mx-mn+1e-6),0,1); nz=-0.5*nz+1.0
    ring_min=torch.minimum(torch.minimum(torch.minimum(b_,d_),f_),h_); ring_max=torch.maximum(torch.maximum(torch.maximum(b_,d_),f_),h_)
    hitMin=torch.minimum(ring_min,e_)/(4.0*ring_max+1e-8); hitMax=(1.0-torch.maximum(ring_max,e_))/(4.0*ring_min-4.0+1e-8)
    lobe_rgb=torch.maximum(-hitMin,hitMax)
    lobe=torch.maximum(torch.tensor(-limit),torch.minimum(lobe_rgb.max(1,keepdim=True).values,torch.tensor(0.0)))*sharpness*nz
    rcpL=1.0/(4.0*lobe+1.0)
    return torch.clamp((lobe*(b_+d_+h_+f_)+e_)*rcpL,0,1)
img=Image.open('D:/work/output/adiff_out/sd15_512.png').convert('RGB')
t=torch.from_numpy(np.asarray(img)).permute(2,0,1).unsqueeze(0).float()/255.0
# 放大到 2048 (4x) 对比: bicubic vs bicubic+RCAS
up=F.interpolate(t,scale_factor=4,mode='bicubic',align_corners=False)
sh=rcas(up,sharpness=1.0)  # 高锐度, 展示 RCAS 缺陷
Image.fromarray((up[0].permute(1,2,0).numpy()*255).astype('uint8')).save('D:/work/output/adiff_out/fsr_defect_bicubic.png')
Image.fromarray((sh[0].permute(1,2,0).numpy()*255).astype('uint8')).save('D:/work/output/adiff_out/fsr_defect_rcas.png')
print('512->2048 (4x): bicubic 存 fsr_defect_bicubic, RCAS锐化 存 fsr_defect_rcas, RSS %.2fGB'%psutil.Process().memory_info().rss/1e9)
