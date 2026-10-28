import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import torch, numpy as np, av, time
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import resize
raft=raft_large(weights=Raft_Large_Weights.DEFAULT); raft.eval()
def flow_to_dense(flow):
    return flow  # B,2,H,W
def warp(img, flow):
    # img B,3,H,W, flow B,2,H,W -> grid_sample warp to flow
    b,c,h,w=img.shape
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    vx=xx.float().unsqueeze(0); vy=yy.float().unsqueeze(0)
    fx=flow[:,0:1]; fy=flow[:,1:2]
    gx=vx+fx; gy=vy+fy
    gx=2.0*gx/(w-1)-1.0; gy=2.0*gy/(h-1)-1.0
    grid=torch.stack([gx,gy],dim=-1)
    return F.grid_sample(img,grid.to(img.device),mode='bilinear',align_corners=True)
# 读视频帧 (512 有内容)
c=av.open('D:/work/output/adiff_out/jnitaimei_512.mp4'); frames=[]
for f in c.decode(video=0): frames.append(f.to_ndarray(format='rgb24'))
c.close()
tens=[torch.from_numpy(fr).permute(2,0,1).unsqueeze(0).float()/255.0 for fr in frames]
t0=time.time(); out_frames=[]
scale=1.4
for i in range(len(tens)):
    # 当前帧 512 -> 720 (bi), 用前后帧光流 warp 累积
    cur=tens[i]; ih=i-1; inc=i+1
    acc=cur; cnt=1
    if ih>=0:
        fl=raft(tens[ih],cur)[-1]; warped=warp(tens[ih],fl); acc=acc+warped; cnt+=1
    if inc<len(tens):
        fl2=raft(cur,tens[inc])[-1]; # 反向
        # 简化: 用当前帧到前一帧的流 warp 前一帧
        acc=acc+warp(cur,fl2); cnt+=1
    avg=acc/cnt
    up=F.interpolate(avg,scale_factor=scale,mode='bicubic',align_corners=False)
    out_frames.append((up[0].permute(1,2,0).numpy()*255).astype('uint8'))
dt=time.time()-t0
print('时序超分(RAFT warp+累积) 8帧 512->%(W)d 耗时 %.2fs (每帧 %.2fs)'%{'W':int(512*scale)}%(dt,dt/len(tens)))
out=av.open('D:/work/output/adiff_out/temp_sr_720.mp4','w'); st=out.add_stream('mpeg4',rate=8); st.width=int(512*scale); st.height=int(512*scale); st.pix_fmt='yuv420p'
for fr in out_frames:
    for p in st.encode(av.VideoFrame.from_ndarray(fr,format='rgb24')): out.mux(p)
for p in st.encode(): out.mux(p)
out.close()
from PIL import Image
Image.fromarray(out_frames[4]).save('D:/work/output/adiff_out/temp_sr_preview.png')
print('已存 temp_sr_720.mp4')
