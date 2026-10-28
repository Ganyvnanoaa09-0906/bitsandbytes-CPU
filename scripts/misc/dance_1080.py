import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys, time, threading
sys.path.insert(0, r"D:\work\那很有乐子了~\bitsandbytes")
import numpy as np, torch, av, psutil
import torch.nn.functional as F
from PIL import Image
BASE="D:/work/textmodel/sd15_base/unet"; FULL="D:/work/textmodel/sd15_full"
torch.set_num_threads(12); torch.backends.mkldnn.enabled=True
from diffusers import UNet2DConditionModel, MotionAdapter, AnimateDiffPipeline, AutoencoderKL, DDPMScheduler
from diffusers.models import UNetMotionModel
from transformers import CLIPTextModel, CLIPTokenizer
def rcas(img, sharpness=0.7, limit=0.25-(1.0/16.0)):
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
unet=UNet2DConditionModel.from_pretrained(BASE,torch_dtype=torch.float32); ma=MotionAdapter.from_pretrained("D:/work/textmodel/animatediff-motion-adapter-v1-5-2"); um=UNetMotionModel.from_unet2d(unet,ma)
vae=AutoencoderKL.from_pretrained(FULL+"/vae",torch_dtype=torch.float32); te=CLIPTextModel.from_pretrained(FULL+"/text_encoder",torch_dtype=torch.float32); tok=CLIPTokenizer.from_pretrained(FULL+"/text_encoder"); sch=DDPMScheduler.from_pretrained(FULL+"/scheduler")
pipe=AnimateDiffPipeline(unet=um,vae=vae,text_encoder=te,tokenizer=tok,scheduler=sch,motion_adapter=ma)
prompt="a young woman dancing gracefully, flowing dress, elegant stage, cinematic, detailed"
peak=[0.0]; stop=[False]
def samp():
    p=psutil.Process()
    while not stop[0]:
        m=p.memory_info().rss/1e9
        if m>peak[0]: peak[0]=m
        time.sleep(0.05)
th=threading.Thread(target=samp); th.start()
t0=time.time()
with torch.no_grad(): out=pipe(prompt=prompt,num_frames=8,num_inference_steps=20,guidance_scale=7.5,height=512,width=512)
gen=time.time()-t0; stop[0]=True; th.join()
print("生成 '小姐姐跳舞' 512x512x8帧 %.1fs 峰值RSS %.2fGB"%(gen,peak[0]))
# FSR 超分 512 -> 1080x1080 (方形) / 或 到 1080p
t0=time.time(); out_frames=[]
for fr in out.frames[0]:
    t=torch.from_numpy(np.asarray(fr)).permute(2,0,1).unsqueeze(0).float()/255.0
    up=F.interpolate(t,size=(1080,1080),mode='bicubic',align_corners=False); sh=rcas(up)
    out_frames.append((sh[0].permute(1,2,0).numpy()*255).astype('uint8'))
fsr=time.time()-t0
print("FSR 512->1080x1080 8帧 %.2fs"%(fsr))
out=av.open('D:/work/output/adiff_out/dance_1080.mp4','w'); st=out.add_stream('mpeg4',rate=8); st.width=1080; st.height=1080; st.pix_fmt='yuv420p'
for fr in out_frames:
    for p in st.encode(av.VideoFrame.from_ndarray(fr,format='rgb24')): out.mux(p)
for p in st.encode(): out.mux(p)
out.close()
Image.fromarray(out_frames[4]).save('D:/work/output/adiff_out/dance_1080_preview.png')
print("已存 dance_1080.mp4 (1080x1080 8帧 小姐姐跳舞)")
