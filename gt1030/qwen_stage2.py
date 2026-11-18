# -*- coding: utf-8 -*-
"""qwen_stage2.py — 两阶段批量的「阶段 B」单独跑：从已落盘的 latent 解码 + 超分 + 存盘。

背景：阶段 A（降噪）已经把 3 个 latent 落盘；阶段 B 第一次运行时整机死机（VAE 分块解码 +
后台超分并发导致内存打满）。这个脚本把阶段 B 改成**严格串行、每步打日志、可选只跑某一张**，
方便逐张验证而不是一口气把机器跑死。

用法：
  python qwen_stage2.py --preset 1080p --only 0            # 只处理第 0 张
  python qwen_stage2.py --preset 1080p                     # 全部（串行）
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import psutil
import torch

from qwen_img import FULL, PRESETS
from sr_models import DEFAULT as DEFAULT_SR
from PIL import Image

LAT = r"D:\work\research\bench_results\qwen_batch2\latents"
OUT = r"D:\work\research\bench_results\qwen_batch2"


def rss():
    return psutil.Process().memory_info().rss / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="1080p", choices=list(PRESETS))
    ap.add_argument("--only", type=int, default=-1, help="只处理这一张（-1 = 全部）")
    ap.add_argument("--sr", default=DEFAULT_SR)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--no-sr", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    gw, gh, up, ow, oh = PRESETS[a.preset]
    idxs = [a.only] if a.only >= 0 else list(range(20))

    from diffusers.models import AutoencoderKLQwenImage21
    t0 = time.time()
    vae = AutoencoderKLQwenImage21.from_pretrained(FULL, subfolder="vae",
                                                   torch_dtype=torch.float32,
                                                   local_files_only=True).eval()
    if max(gw, gh) >= 1024 or max(ow, oh) >= 1024:
        vae.enable_tiling()
    print(f"  VAE 载入 {time.time()-t0:.1f}s | tiling | RSS {rss():.2f} GB", flush=True)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    done = []
    for i in idxs:
        p = os.path.join(LAT, f"lat_{i}.npy")
        if not os.path.exists(p):
            continue
        lat = torch.from_numpy(np.load(p))
        seq_dim, ch = lat.shape[1], lat.shape[-1]     # (1, seq, ch)
        h_lat, w_lat = gh // 16, gw // 16
        t1 = time.time()
        z = lat.reshape(1, seq_dim, ch).transpose(1, 2).reshape(1, ch, 1, h_lat, w_lat)
        with torch.no_grad():
            img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0][:, :3]
        arr = ((img.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).permute(0, 2, 3, 1)[0].numpy()
        vae_s = time.time() - t1
        del img, z, lat
        gc.collect()
        print(f"  [{i}] VAE 解码 {vae_s:.1f}s | {arr.shape} | RSS {rss():.2f} GB", flush=True)

        sr_s = 0.0
        if not a.no_sr and a.sr:
            from sr_models import upscale_array
            t2 = time.time()
            arr2 = upscale_array(arr, name=a.sr, out_w=ow, out_h=oh)[0]
            sr_s = time.time() - t2
            print(f"  [{i}] 超分 {sr_s:.1f}s → {arr2.shape} | RSS {rss():.2f} GB", flush=True)
        else:
            arr2 = arr
        out = os.path.join(OUT, f"stage2_{a.preset}_{a.tag}_{i}.png")
        Image.fromarray(arr2).save(out)
        done.append(dict(idx=i, path=out, vae_s=round(vae_s, 1), sr_s=round(sr_s, 1),
                         shape=list(arr2.shape)))
        del arr, arr2
        gc.collect()
        print(f"  [{i}] 已保存 {out} | RSS {rss():.2f} GB", flush=True)

    print("RESULT " + json.dumps(done, ensure_ascii=False))


if __name__ == "__main__":
    main()
