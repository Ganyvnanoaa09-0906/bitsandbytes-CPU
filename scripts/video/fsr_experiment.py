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
import torch, time, av, numpy as np, psutil
import torch.nn.functional as F
def rcas(img, sharpness=0.8, limit=0.25-(1.0/16.0)):
    b,c,h,w = img.shape
    pad = F.pad(img, (1,1,1,1), mode='replicate')
    b_=pad[:,:,:-2,1:-1]; d_=pad[:,:,2:,1:-1]; f_=pad[:,:,1:-1,:-2]; h_=pad[:,:,1:-1,2:]; e_=img
    def lum(x): return 0.5*x[:,2:3]+(0.5*x[:,0:1]+x[:,1:2])
    bL,dL,eL,fL,hL=lum(b_),lum(d_),lum(e_),lum(f_),lum(h_)
    nz=0.25*(bL+dL+fL+hL)-eL
    mx=torch.maximum(torch.maximum(torch.maximum(bL,dL),eL),torch.maximum(fL,hL))
    mn=torch.minimum(torch.minimum(torch.minimum(bL,dL),eL),torch.minimum(fL,hL))
    nz=torch.clamp(torch.abs(nz)/(mx-mn+1e-6),0,1); nz=-0.5*nz+1.0
    ring_min=torch.minimum(torch.minimum(torch.minimum(b_,d_),f_),h_); ring_max=torch.maximum(torch.maximum(torch.maximum(b_,d_),f_),h_)
    hitMin=torch.minimum(ring_min,e_)/(4.0*ring_max+1e-8); hitMax=(1.0-torch.maximum(ring_max,e_))/(4.0*ring_min-4.0+1e-8)
    lobe_rgb=torch.maximum(-hitMin,hitMax)
    lobe=torch.maximum(torch.tensor(-limit),torch.minimum(lobe_rgb.max(1,keepdim=True).values,torch.tensor(0.0)))*sharpness*nz
    rcpL=1.0/(4.0*lobe+1.0)
    return torch.clamp((lobe*(b_+d_+h_+f_)+e_)*rcpL,0,1)
# 读视频帧
c=av.open('D:/work/output/adiff_out/jnitaimei_512.mp4'); frames=[]
for f in c.decode(video=0): frames.append(f.to_ndarray(format='rgb24'))
c.close()
print('视频帧数', len(frames), '每帧', frames[0].shape)
# 逐帧 FSR 放大 (512x512 -> 2048x2048, bicubic + RCAS)
scale=4; t0=time.time()
out_frames=[]
for fr in frames:
    t=torch.from_numpy(fr).permute(2,0,1).unsqueeze(0).float()/255.0
    up=F.interpolate(t,scale_factor=scale,mode='bicubic',align_corners=False)
    sh=rcas(up)
    out_frames.append((sh[0].permute(1,2,0).numpy()*255).astype('uint8'))
dt=time.time()-t0
print('FSR 放大 %d帧 512->%dx%d 总耗时 %.2fs (每帧 %.2fs) RSS %.2fGB'%(len(frames),512*scale,512*scale,dt,dt/len(frames),psutil.Process().memory_info().rss/1e9))
# 保存放大视频
out=av.open('D:/work/output/adiff_out/fsr_upscaled_2048.mp4','w'); st=out.add_stream('mpeg4',rate=8); st.width=512*scale; st.height=512*scale; st.pix_fmt='yuv420p'
for fr in out_frames:
    for p in st.encode(av.VideoFrame.from_ndarray(fr,format='rgb24')): out.mux(p)
for p in st.encode(): out.mux(p)
out.close()
print('已存 fsr_upscaled_2048.mp4 (%d帧 2048x2048)'%len(out_frames))
