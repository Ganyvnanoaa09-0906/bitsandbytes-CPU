import os, time, sys, threading
sys.path.insert(0, r"D:\work\那很有乐子了~\bitsandbytes")
import numpy as np, torch, av, psutil
import torch.nn.functional as F
from PIL import Image
BASE="D:/work/textmodel/sd15_base/unet"; FULL="D:/work/textmodel/sd15_full"
torch.set_num_threads(12); torch.backends.mkldnn.enabled=True
from diffusers import UNet2DConditionModel, MotionAdapter, AnimateDiffPipeline, AutoencoderKL, DDPMScheduler
from diffusers.models import UNetMotionModel
from transformers import CLIPTextModel, CLIPTokenizer
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
unet=UNet2DConditionModel.from_pretrained(BASE,torch_dtype=torch.float32); ma=MotionAdapter.from_pretrained("D:/work/textmodel/animatediff-motion-adapter-v1-5-2"); um=UNetMotionModel.from_unet2d(unet,ma)
vae=AutoencoderKL.from_pretrained(FULL+"/vae",torch_dtype=torch.float32); te=CLIPTextModel.from_pretrained(FULL+"/text_encoder",torch_dtype=torch.float32); tok=CLIPTokenizer.from_pretrained(FULL+"/text_encoder"); sch=DDPMScheduler.from_pretrained(FULL+"/scheduler")
pipe=AnimateDiffPipeline(unet=um,vae=vae,text_encoder=te,tokenizer=tok,scheduler=sch,motion_adapter=ma)
prompt="a beautiful woman, cinematic, realistic"
TOTAL_SEG=15; SEG_FRAMES=8; SCALE=16  # 64 -> 1024 (16x)
gen_t=0; fsr_t=0; all_frames=[]
for seg in range(TOTAL_SEG):
    t0=time.time()
    with torch.no_grad(): out=pipe(prompt=prompt,num_frames=SEG_FRAMES,num_inference_steps=15,guidance_scale=7.5,height=64,width=64)
    seg_frames=out.frames[0]; gen_t+=time.time()-t0
    t0=time.time()
    for fr in seg_frames:
        t=torch.from_numpy(np.asarray(fr)).permute(2,0,1).unsqueeze(0).float()/255.0
        up=F.interpolate(t,scale_factor=SCALE,mode='bicubic',align_corners=False); sh=rcas(up)
        all_frames.append((sh[0].permute(1,2,0).numpy()*255).astype('uint8'))
    fsr_t+=time.time()-t0
    print("[seg %d/%d] 生成%.1fs + FSR%.1fs 累计帧%d"%(seg+1,TOTAL_SEG, (time.time()-t0), fsr_t, len(all_frames)), flush=True)
print("生成%.1fs FSR%.1fs 总帧%d"%(gen_t,fsr_t,len(all_frames)))
out=av.open('D:/work/output/adiff_out/fsr_15s.mp4','w'); st=out.add_stream('mpeg4',rate=8); st.width=1024; st.height=1024; st.pix_fmt='yuv420p'
for fr in all_frames:
    for p in st.encode(av.VideoFrame.from_ndarray(fr,format='rgb24')): out.mux(p)
for p in st.encode(): out.mux(p)
out.close()
print("已存 fsr_15s.mp4 (%d帧 1024x1024 约15秒)"%len(all_frames))
