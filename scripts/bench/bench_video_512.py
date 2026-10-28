"""Benchmark AnimateDiff video generation speed at various resolutions/schedulers."""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse
import os
import sys
import time
import threading
import psutil

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# make sure local bnb is used if needed
if os.path.isdir(os.path.join(_HERE, "bitsandbytes")):
    sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

BASE = "D:/work/textmodel/sd15_base/unet"
FULL = "D:/work/textmodel/sd15_full"
ADAPTER = "D:/work/textmodel/animatediff-motion-adapter-v1-5"


def build_pipe(scheduler: str = "ddim", adapter: str = ADAPTER):
    import torch
    from diffusers import (
        AnimateDiffPipeline,
        AutoencoderKL,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        MotionAdapter,
        UNet2DConditionModel,
        PNDMScheduler,
    )
    from diffusers.models import UNetMotionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    unet = UNet2DConditionModel.from_pretrained(BASE, torch_dtype=torch.float32)
    ma = MotionAdapter.from_pretrained(adapter, torch_dtype=torch.float32)
    um = UNetMotionModel.from_unet2d(unet, ma)
    vae = AutoencoderKL.from_pretrained(FULL + "/vae", torch_dtype=torch.float32)
    te = CLIPTextModel.from_pretrained(FULL + "/text_encoder", torch_dtype=torch.float32)
    tok = CLIPTokenizer.from_pretrained(FULL + "/text_encoder")
    if scheduler == "ddim":
        sch = DDIMScheduler.from_pretrained(FULL + "/scheduler", subfolder=None)
    elif scheduler == "dpm":
        sch = DPMSolverMultistepScheduler.from_pretrained(FULL + "/scheduler")
    elif scheduler == "euler":
        sch = EulerAncestralDiscreteScheduler.from_pretrained(FULL + "/scheduler")
    else:
        sch = PNDMScheduler.from_pretrained(FULL + "/scheduler")
    pipe = AnimateDiffPipeline(
        unet=um, vae=vae, text_encoder=te, tokenizer=tok,
        scheduler=sch, motion_adapter=ma,
    )
    return pipe


def mem_peak_monitor(peak, stop):
    p = psutil.Process()
    while not stop.is_set():
        try:
            m = p.memory_info().rss / 1e9
            if m > peak[0]:
                peak[0] = m
        except Exception:
            pass
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--scheduler", default="dpm")
    ap.add_argument("--prompt", default="a beautiful woman, cinematic, realistic, detailed face")
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True

    print(f"[bench] threads={a.threads} scheduler={a.scheduler} res={a.res} "
          f"frames={a.frames} steps={a.steps} adapter={os.path.basename(a.adapter)}", flush=True)

    t_load = time.time()
    pipe = build_pipe(a.scheduler, a.adapter)
    print(f"[bench] load {time.time()-t_load:.1f}s", flush=True)

    peak = [0.0]
    stop = threading.Event()
    th = threading.Thread(target=mem_peak_monitor, args=(peak, stop), daemon=True)
    th.start()

    t0 = time.time()
    with torch.no_grad():
        out = pipe(
            prompt=a.prompt,
            num_frames=a.frames,
            num_inference_steps=a.steps,
            guidance_scale=7.5,
            height=a.res,
            width=a.res,
        )
    dt = time.time() - t0
    stop.set()
    th.join(timeout=0.2)
    frames = out.frames[0]
    print(f"[bench] RESULT {a.res}x{a.res}x{len(frames)}f x{a.steps}step = {dt:.2f}s "
          f"(per-frame {dt/len(frames):.2f}s, per-frame-step {dt/(len(frames)*a.steps):.3f}s) "
          f"RSS_peak {peak[0]:.2f}GB", flush=True)

    if not a.no_save:
        import numpy as np
        from PIL import Image
        os.makedirs("D:/work/output/adiff_out", exist_ok=True)
        imgs = [Image.fromarray(np.asarray(f)).convert("RGB") for f in frames]
        tag = f"bench_{a.scheduler}_{a.res}_{a.frames}f_{a.steps}s"
        imgs[0].save(f"D:/work/output/adiff_out/{tag}.gif", save_all=True,
                     append_images=imgs[1:], duration=200, loop=0)
        for i in [0, len(imgs)//2, len(imgs)-1]:
            imgs[i].save(f"D:/work/output/adiff_out/{tag}_f{i}.png")
        print(f"[bench] saved {tag}.gif")


if __name__ == "__main__":
    main()
