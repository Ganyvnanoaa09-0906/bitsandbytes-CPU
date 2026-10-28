"""Generate multiple low-res AnimateDiff clips in one process.

Outputs PNG frames to out_dir, one file per frame, named seg000_f000.png ...
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse
import os
import sys
import time
import threading
import numpy as np
import psutil
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
if os.path.isdir(os.path.join(_HERE, "bitsandbytes")):
    sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

BASE = "D:/work/textmodel/sd15_base/unet"
FULL = "D:/work/textmodel/sd15_full"
ADAPTER = "D:/work/textmodel/animatediff-motion-adapter-v1-5"


def build_pipe(scheduler="dpm", adapter=ADAPTER):
    import torch
    from diffusers import (
        AnimateDiffPipeline,
        AutoencoderKL,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        MotionAdapter,
        UNet2DConditionModel,
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
        sch = DDIMScheduler.from_pretrained(FULL + "/scheduler")
    else:
        sch = DPMSolverMultistepScheduler.from_pretrained(FULL + "/scheduler")
    pipe = AnimateDiffPipeline(unet=um, vae=vae, text_encoder=te, tokenizer=tok,
                               scheduler=sch, motion_adapter=ma)
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
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--scheduler", default="dpm")
    ap.add_argument("--out", default=r"D:\work\output\adiff_out\clips_128_8f_3s")
    ap.add_argument("--prompt", default="a beautiful woman, cinematic, realistic, detailed face")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True

    os.makedirs(a.out, exist_ok=True)
    # clean old frames
    for f in os.listdir(a.out):
        if f.endswith(".png"):
            os.remove(os.path.join(a.out, f))

    print(f"[gen] res={a.res} frames={a.frames} steps={a.steps} segments={a.segments} "
          f"threads={a.threads} scheduler={a.scheduler}", flush=True)
    t_load = time.time()
    pipe = build_pipe(a.scheduler)
    print(f"[gen] load {time.time()-t_load:.1f}s", flush=True)

    peak = [0.0]
    stop = threading.Event()
    th = threading.Thread(target=mem_peak_monitor, args=(peak, stop), daemon=True)
    th.start()

    all_frames = []
    total0 = time.time()
    for seg in range(a.segments):
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
        frames = out.frames[0]
        dt = time.time() - t0
        all_frames.extend(frames)
        # save immediately
        for fi, fr in enumerate(frames):
            im = Image.fromarray(np.asarray(fr)).convert("RGB")
            im.save(os.path.join(a.out, f"seg{seg:03d}_f{fi:02d}.png"))
        print(f"[gen] seg {seg+1}/{a.segments} {dt:.1f}s (total {time.time()-total0:.1f}s)", flush=True)

    stop.set()
    th.join(timeout=0.2)
    print(f"[gen] DONE frames={len(all_frames)} total={time.time()-total0:.1f}s "
          f"RSS_peak={peak[0]:.2f}GB -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
