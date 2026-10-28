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
import os, sys, time, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"D:\work\那很有乐子了~\bitsandbytes")
import numpy as np, torch, psutil
from PIL import Image
BASE="D:/work/textmodel/sd15_base/unet"; FULL="D:/work/textmodel/sd15_full"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--res",type=int,default=256); ap.add_argument("--frames",type=int,default=8); ap.add_argument("--steps",type=int,default=15); ap.add_argument("--threads",type=int,default=12); a=ap.parse_args()
    torch.set_num_threads(a.threads); torch.backends.mkldnn.enabled=True
    from diffusers import UNet2DConditionModel, MotionAdapter, AnimateDiffPipeline, AutoencoderKL, DDPMScheduler
    from diffusers.models import UNetMotionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    unet=UNet2DConditionModel.from_pretrained(BASE,torch_dtype=torch.float32); ma=MotionAdapter.from_pretrained("D:/work/textmodel/animatediff-motion-adapter-v1-5-2"); um=UNetMotionModel.from_unet2d(unet,ma)
    vae=AutoencoderKL.from_pretrained(FULL+"/vae",torch_dtype=torch.float32); te=CLIPTextModel.from_pretrained(FULL+"/text_encoder",torch_dtype=torch.float32); tok=CLIPTokenizer.from_pretrained(FULL+"/text_encoder"); sch=DDPMScheduler.from_pretrained(FULL+"/scheduler")
    pipe=AnimateDiffPipeline(unet=um,vae=vae,text_encoder=te,tokenizer=tok,scheduler=sch,motion_adapter=ma)
    t0=time.time()
    with torch.no_grad(): out=pipe(prompt="a beautiful woman, realistic, cinematic",num_frames=a.frames,num_inference_steps=a.steps,guidance_scale=7.5,height=a.res,width=a.res)
    dt=time.time()-t0; m=psutil.Process().memory_info().rss/1e9
    print("[cpu] %dx%dx%dframe x%ds = %.1fs (每帧 %.1fs) RSS %.2fGB"%(a.res,a.res,a.frames,a.steps,dt,dt/a.frames,m))
    os.makedirs("D:/work/output/adiff_out",exist_ok=True)
    imgs=[Image.fromarray(np.asarray(f)).convert("RGB") for f in out.frames[0]]
    imgs[0].save("D:/work/output/adiff_out/cpu2.gif",save_all=True,append_images=imgs[1:],duration=200,loop=0)
if __name__=="__main__": main()
