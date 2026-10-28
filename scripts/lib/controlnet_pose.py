# -*- coding: utf-8 -*-
"""ControlNet (OpenPose) guidance for the AnimateDiff pipeline, implemented by hand.

diffusers 0.27 ships no AnimateDiff+ControlNet pipeline, so the ControlNet residuals are
injected into our own sampling loop:

    down_res, mid_res = controlnet(flat_latents, t, encoder_hidden_states=embeds,
                                   controlnet_cond=control_frames)
    noise = unet(latents5d, t, encoder_hidden_states=embeds,
                 down_block_additional_residuals=down_res,
                 mid_block_additional_residual=mid_res)

The UNetMotionModel wants 5D (B,C,T,H,W) while ControlNetModel is a 2D UNet, so the frame axis
is folded into the batch for the ControlNet call and the residuals are unfolded back.

Skeletons come from a real frame (keypointrcnn, COCO-17 -> OpenPose-18) and are then
ANIMATED with sinusoidal limb swings, because the collapsed L4/L5 frames cannot supply a
usable pose sequence (det% 80/40) -- the point of the experiment is to ask whether giving the
model a *correct* skeleton rescues the anatomy.

Usage:
  python controlnet_pose.py --pose-src <frames dir> --pose-frame 5 \
      --prompt "..." --out <dir> --chunk 10 --steps 5
"""
import os, sys
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
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from anime_adiff import build, ANIME_NEG, OUT_DIR, VAE_DIR  # noqa: E402

GUIDANCE = 7.5
CN_DIR = r"D:\work\textmodel\controlnet_openpose"

# OpenPose-18 limb pairs (index space below) and the standard BGR palette
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
PALETTE = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
           (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
           (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
           (255, 0, 255), (255, 0, 170), (255, 0, 85)]

# COCO-17 index -> OpenPose-18 index
COCO2OP = {0: 0, 5: 2, 6: 5, 7: 3, 8: 6, 9: 4, 10: 7, 11: 8, 12: 11,
           13: 9, 14: 12, 15: 10, 16: 13, 1: 15, 2: 14, 3: 17, 4: 16}


def get_pose_model():
    from torchvision.models.detection import (keypointrcnn_resnet50_fpn,
                                              KeypointRCNN_ResNet50_FPN_Weights)
    m = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    m.eval()
    return m


def extract_openpose18(model, img_path, thresh=0.4):
    """Return 18 OpenPose keypoints [18,3] = (x, y, conf) in pixel coords, or None."""
    arr = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = arr.shape[:2]
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = model([t])[0]
    sc = out["scores"].numpy()
    if len(sc) == 0 or sc.max() < thresh:
        return None
    i = int(sc.argmax())
    kp = out["keypoints"][i].numpy()          # [17,3]
    kps = out["keypoints_scores"][i].numpy()  # [17]
    conf = np.clip(kps / 10.0, 0, 1)          # keypointrcnn scores -> 0..1
    op = np.zeros((18, 3), dtype=np.float32)
    for c, o in COCO2OP.items():
        op[o, 0] = kp[c, 0]
        op[o, 1] = kp[c, 1]
        op[o, 2] = conf[c]
    # neck = midpoint of shoulders (OpenPose index 1); requires both shoulders
    if op[2, 2] > 0 and op[5, 2] > 0:
        op[1, 0] = (op[2, 0] + op[5, 0]) / 2
        op[1, 1] = (op[2, 1] + op[5, 1]) / 2
        op[1, 2] = min(op[2, 2], op[5, 2])
    return op, (W, H)


def _rot(p, o, ang):
    """Rotate point p about origin o by `ang` radians, preserving segment length."""
    rel = np.asarray(p, dtype=np.float32) - np.asarray(o, dtype=np.float32)
    ca, sa = np.cos(ang), np.sin(ang)
    return np.asarray(o, dtype=np.float32) + np.array([rel[0] * ca - rel[1] * sa,
                                                       rel[0] * sa + rel[1] * ca],
                                                      dtype=np.float32)


def animate_pose(op, frame, n_frames, amp=1.0):
    """Turn one detected pose into an articulated run cycle.

    v1 used sin(2*pi*i/n), which is zero at BOTH i=0 and i=n/2, so frames 0 and 5 came out
    with an identical skeleton. v2 uses cos(), so frame 0 and frame n/2 are the two opposite
    extremes, and the limbs are rotated as a proper 2-segment chain (elbow/knee first, then
    the forearm/shin about the *new* joint) so limb lengths never stretch.
    """
    k = np.array(op, dtype=np.float32, copy=True)
    ph = 2 * np.pi * frame / max(n_frames, 1)
    # A pure cos() cycle is a palindrome: pose(i) == pose(n-i), so half the frames are
    # duplicates. Mixing in the 2nd harmonic (which flips sign under i -> n-i) gives every
    # frame a distinct skeleton while keeping the swing smooth.
    c = (float(np.cos(ph)) + 0.30 * float(np.sin(2 * ph))) * amp
    cl = (float(np.cos(ph)) - 0.30 * float(np.sin(2 * ph))) * amp
    # scale limb motion by torso size so it works at any framing
    sh = (k[2, :2] + k[5, :2]) / 2
    hip = (k[8, :2] + k[11, :2]) / 2
    torso = float(np.linalg.norm(sh - hip)) or 50.0

    # arms: contralateral swing (right arm forward while left arm back), about the shoulder.
    # The wrist is first carried along by the SAME rotation as the upper arm (that is what
    # keeps the forearm rigid), and only then bent a little further about the new elbow.
    for wrist, elbow, sh_i, s in ((4, 3, 2, 1), (7, 6, 5, -1)):
        if k[sh_i, 2] > 0:
            ang = 0.55 * c * s
            carried = _rot(k[wrist, :2], k[sh_i, :2], ang)
            ne = _rot(k[elbow, :2], k[sh_i, :2], ang)
            k[wrist, :2] = _rot(carried, ne, ang * 0.7)
            k[elbow, :2] = ne
    # legs: opposite to the same-side arm, about the hip
    for knee, ankle, hip_i, s in ((9, 10, 8, 1), (12, 13, 11, -1)):
        if k[hip_i, 2] > 0:
            ang = -0.50 * cl * s
            carried = _rot(k[ankle, :2], k[hip_i, :2], ang)
            nk = _rot(k[knee, :2], k[hip_i, :2], ang)
            k[ankle, :2] = _rot(carried, nk, ang * 0.7)
            k[knee, :2] = nk
    # whole body bobs vertically at twice the limb frequency (two steps per cycle).
    # The floor matters: when the source frame is itself collapsed the detected torso can be
    # only ~12px, which would make the bob invisible.
    bob = 0.10 * max(torso, 80.0) * (np.cos(2 * ph) + 0.25 * np.sin(ph)) * amp
    k[:, 1] += bob
    # head follows the torso slightly
    if k[1, 2] > 0:
        off = 0.05 * max(torso, 80.0) * c
        for idx in (0, 14, 15, 16, 17):
            k[idx, 0] += off
    return k


def render_openpose(op, size):
    """Draw an OpenPose-style skeleton (limbs + joints) on black."""
    W, H = size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    for li, (a, b) in enumerate(LIMBS):
        if op[a, 2] > 0 and op[b, 2] > 0:
            dr.line([tuple(op[a, :2].tolist()), tuple(op[b, :2].tolist())],
                    fill=PALETTE[li % len(PALETTE)], width=max(3, H // 128))
    for j in range(18):
        if op[j, 2] > 0:
            x, y = op[j, :2]
            r = max(3, H // 110)
            dr.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-src", required=True, help="frames dir to take the base pose from")
    ap.add_argument("--pose-frame", type=int, default=5)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--neg", default=ANIME_NEG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cn-scale", type=float, default=1.0)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--lora", nargs="*", default=[],
                    help="zero or more LoRA dirs/files, merged into the UNet before sampling")
    ap.add_argument("--lora-scale", nargs="*", type=float, default=[],
                    help="scale per LoRA, positionally matched (default 0.8 each)")
    ap.add_argument("--no-control", action="store_true",
                    help="run the identical loop WITHOUT controlnet (A/B baseline)")
    ap.add_argument("--ab", action="store_true",
                    help="run control ON and OFF back to back in one process (same models, "
                         "same seed) -> <out>/on and <out>/off")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    os.makedirs(a.out, exist_ok=True)
    pdir = os.path.join(a.out, "poses")
    os.makedirs(pdir, exist_ok=True)

    # ---- 1. base pose from a real frame, then animate a sequence ----
    pm = get_pose_model()
    src = os.path.join(a.pose_src, f"f{a.pose_frame:03d}.png")
    got = extract_openpose18(pm, src)
    if got is None:
        print(f"[err] no person detected in {src}")
        return
    base, (W, H) = got
    print(f"[pose] base pose from {os.path.basename(src)} ({W}x{H}), "
          f"{int((base[:,2]>0).sum())}/18 joints confident")

    poses = [animate_pose(base, i, a.chunk, amp=a.amp) for i in range(a.chunk)]
    ctrl = []
    for i, p in enumerate(poses):
        img = render_openpose(p, (W, H))
        Image.fromarray(img).save(os.path.join(pdir, f"pose_{i:03d}.png"))
        ctrl.append(img)
    print(f"[pose] rendered {len(ctrl)} animated skeletons -> {pdir}")

    # ---- 2. pipeline ----
    pipe = build("dpm")
    if a.lora:
        from lora_merge import merge_loras
        scales = list(a.lora_scale)
        while len(scales) < len(a.lora):
            scales.append(scales[-1] if scales else 0.8)
        merge_loras(pipe, a.lora, scales)
    cn = None
    if not a.no_control:
        from diffusers import ControlNetModel
        print(f"[load] ControlNet from {CN_DIR}", flush=True)
        cn = ControlNetModel.from_pretrained(CN_DIR, torch_dtype=torch.float32)
        cn.eval()

    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.neg)
    embeds = torch.cat([ne, pe])                 # (2,S,D) == (uncond, cond)

    # control tensors: one per frame, folded to match the flattened batch
    ctrl_t = torch.stack([
        torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float() / 255.0
        for c in ctrl
    ])                                           # [T,3,H,W]

    lat_h = a.res // pipe.vae_scale_factor

    def sample(use_cn):
        sched = pipe.scheduler
        sched.set_timesteps(a.steps)
        sched._step_index = 0
        sched.model_outputs = [None] * sched.config.solver_order
        sched._lower_order_nums = 0
        ts = list(sched.timesteps)
        gen = torch.Generator().manual_seed(a.seed)
        lat = torch.randn((1, pipe.unet.config.in_channels, a.chunk, lat_h, lat_h),
                          generator=gen, dtype=torch.float32) * sched.init_noise_sigma
        print(f"[gen] chunk={a.chunk} steps={a.steps} "
              f"controlnet={'OFF' if not use_cn else a.cn_scale}", flush=True)
        step_times = []
        with torch.no_grad():
            for si, t in enumerate(ts):
                st = time.time()
                mi = torch.cat([lat] * 2)                    # (2,C,T,H,W)
                mi = sched.scale_model_input(mi, t)
                kw = {}
                if use_cn:
                    b, c, T, h, w = mi.shape
                    flat = mi.permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
                    cond = ctrl_t.repeat(2, 1, 1, 1)         # uncond+cond share the pose
                    dr, mr = cn(flat, t, encoder_hidden_states=embeds.repeat_interleave(T, 0),
                                controlnet_cond=cond, conditioning_scale=a.cn_scale,
                                return_dict=False)
                    # UNetMotionModel does `sample.permute(0,2,1,3,4).reshape(B*T,...)`
                    # (line 848) and adds residuals to that 4D tensor, so they must stay
                    # 4D in exactly the [b0t0..b0tT-1, b1t0..] order the ControlNet saw.
                    kw = dict(down_block_additional_residuals=list(dr),
                              mid_block_additional_residual=mr)
                npr = pipe.unet(mi, t, encoder_hidden_states=embeds, **kw).sample
                nu, nt = npr.chunk(2)
                npr = nu + GUIDANCE * (nt - nu)
                lat = sched.step(npr, t, lat,
                                 **pipe.prepare_extra_step_kwargs(gen, 0.0)).prev_sample
                step_times.append(time.time() - st)
                print(f"    step {si+1}/{len(ts)} t={int(t)}  {step_times[-1]:.1f}s", flush=True)

        with torch.no_grad():
            b, c, T, h, w = lat.shape
            flat = (lat / pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4
                                                                  ).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            arr = (((dec / 2 + 0.5).clamp(0, 1)).reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                   .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype(np.uint8)
        return arr, step_times

    runs = [(True, os.path.join(a.out, "on")), (False, os.path.join(a.out, "off"))] if a.ab \
        else [((cn is not None), a.out)]

    for use_cn, d in runs:
        t0 = time.time()
        arr, step_times = sample(use_cn)
        fdir = os.path.join(d, "frames")
        os.makedirs(fdir, exist_ok=True)
        for i, f in enumerate(arr):
            Image.fromarray(f).save(os.path.join(fdir, f"f{i:03d}.png"))
        dt = time.time() - t0
        print(f"\n[RESULT] {'CONTROL' if use_cn else 'NOCTRL '} {len(arr)} frames in {dt:.0f}s  "
              f"({np.mean(step_times):.1f}s/step avg, {np.max(step_times):.1f}s max)")
        print(f"[save] {fdir}\n", flush=True)


if __name__ == "__main__":
    main()

def extract_openpose18_multi(model, img_path, thresh=0.4, max_persons=2):
    """Return list of (op18, (W,H), score) for up to max_persons people."""
    arr = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = arr.shape[:2]
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = model([t])[0]
    sc = out["scores"].numpy()
    if len(sc) == 0:
        return []
    order = np.argsort(-sc)[:max_persons]
    res = []
    for idx in order:
        if sc[idx] < thresh:
            continue
        kp = out["keypoints"][idx].numpy()
        kps = out["keypoints_scores"][idx].numpy()
        conf = np.clip(kps / 10.0, 0, 1)
        op = np.zeros((18, 3), dtype=np.float32)
        for c, o in COCO2OP.items():
            op[o, 0] = kp[c, 0]
            op[o, 1] = kp[c, 1]
            op[o, 2] = conf[c]
        if op[2, 2] > 0 and op[5, 2] > 0:
            op[1, 0] = (op[2, 0] + op[5, 0]) / 2
            op[1, 1] = (op[2, 1] + op[5, 1]) / 2
            op[1, 2] = min(op[2, 2], op[5, 2])
        res.append((op, (W, H), float(sc[idx])))
    return res

def render_openpose_multi(persons, size, colors=None):
    """Draw multiple OpenPose-18 skeletons with different colors."""
    W, H = size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    colors = colors or [(255, 0, 0), (0, 128, 255), (0, 220, 0), (255, 0, 255)]
    for pi, (op, _, _) in enumerate(persons):
        base = colors[pi % len(colors)]
        for li, (a, b) in enumerate(LIMBS):
            if op[a, 2] > 0 and op[b, 2] > 0:
                c = tuple(int(v * (0.55 + 0.45 * (li % 3) / 2)) for v in base)
                dr.line([tuple(op[a, :2].tolist()), tuple(op[b, :2].tolist())],
                        fill=c, width=max(3, H // 128))
        for j in range(18):
            if op[j, 2] > 0:
                x, y = op[j, :2]
                r = max(3, H // 110)
                dr.ellipse([x - r, y - r, x + r, y + r], fill=base)
    return np.asarray(im)