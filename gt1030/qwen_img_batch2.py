# -*- coding: utf-8 -*-
"""qwen_img_batch2.py — 两阶段批量出图（解决 1080p 批量 OOM）。

阶段 A：只驻留 transformer，逐张降噪，把 latent 落盘（不加载 VAE、不驻留第二份权重）
阶段 B：释放 transformer，只驻留 VAE，逐张解码 + 超分 + 存盘（解码时后台线程可并行超分）

这样峰值内存不再同时包含「transformer + VAE + 双份高分辨率激活」，1080p 也能批量跑。

用法：
  python qwen_img_batch2.py --preset 1080p --prompts "a cat" "a fox" --igpu none
  python qwen_img_batch2.py --preset 1024 --prompts ... --igpu fp16 --overlap-sr 1
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import threading
import time

import numpy as np
import psutil
import torch
from PIL import Image

from qwen4bit_loader import load_transformer
from qwen_lightning_bench import LORA, LoRAInjector, calculate_shift
from qwen_img import FULL, PRESETS, VAE_SCALE, prompt_embeds
from sr_models import DEFAULT as DEFAULT_SR

OUT = r"D:\work\research\bench_results\qwen_batch2"
LAT = r"D:\work\research\bench_results\qwen_batch2\latents"


def rss():
    return psutil.Process().memory_info().rss / 1e9


def z_dim():
    with open(os.path.join(FULL, "vae", "config.json"), encoding="utf-8") as f:
        return int(json.load(f)["z_dim"])


def stage_a(prompts, gen_w, gen_h, steps, sigmas, seed, igpu, threads=6):
    """只跑降噪，返回 latent 文件列表与耗时。"""
    torch.set_num_threads(threads)
    os.makedirs(LAT, exist_ok=True)
    t0 = time.perf_counter()
    tr = load_transformer()
    if igpu != "none":
        from qwen4bit_igpu_linear_fp32 import to_igpu_f32
        to_igpu_f32(tr, mode_fn=(lambda n: "fp16") if igpu == "fp16" else None)
    inj = LoRAInjector(tr, LORA, scale=1.0)
    load_s = time.perf_counter() - t0
    h_lat, w_lat = gen_h // VAE_SCALE, gen_w // VAE_SCALE
    ch = z_dim()

    from diffusers import FlowMatchEulerDiscreteScheduler
    rows = []
    for i, p in enumerate(prompts):
        embeds = prompt_embeds(p)
        g = torch.Generator().manual_seed(seed + i)
        lat = torch.randn(1, 1, ch, h_lat, w_lat, generator=g).float()
        lat = lat.view(1, ch, h_lat * w_lat).transpose(1, 2)
        seq = lat.shape[1]
        pe = embeds["prompt_embeds"].float()
        pm = embeds.get("prompt_embeds_mask")
        base = embeds.get("image_pad_mask", torch.zeros(1, pe.shape[1], dtype=torch.bool)).bool()
        img_mask = torch.cat([base, torch.ones(1, seq // 4, dtype=torch.bool)], dim=1)
        shapes = [[(1, h_lat, w_lat)]]
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(FULL, subfolder="scheduler",
                                                               local_files_only=True)
        sched.set_timesteps(sigmas=sigmas, mu=calculate_shift(seq), device="cpu")
        ts = sched.timesteps
        t1 = time.perf_counter()
        with torch.no_grad():
            for t in ts:
                pred = tr(hidden_states=lat, timestep=(t.expand(1).to(lat.dtype) / 1000.0),
                          encoder_hidden_states=pe, encoder_hidden_states_mask=pm,
                          img_shapes=shapes, img_mask=img_mask, return_dict=False)[0][:, -seq:]
                lat = sched.step(pred, t, lat, return_dict=False)[0]
        dn = time.perf_counter() - t1
        path = os.path.join(LAT, f"lat_{i}.npy")
        np.save(path, lat.detach().numpy().astype(np.float32))
        rows.append(dict(idx=i, prompt=p, denoise_s=round(dn, 1), latent=path,
                         shape=[ch, h_lat, w_lat]))
        print(f"  [A] {i} 降噪 {dn:.1f}s → {os.path.basename(path)} | RSS {rss():.2f} GB", flush=True)

    inj.remove()
    del tr, inj
    gc.collect()
    print(f"  [A] transformer 已释放 | RSS {rss():.2f} GB | 阶段总 {time.perf_counter()-t0:.1f}s", flush=True)
    return dict(rows=rows, load_s=round(load_s, 1), stage_s=round(time.perf_counter() - t0, 1),
                gen=[gen_w, gen_h], ch=ch, h_lat=h_lat, w_lat=w_lat)


def stage_b(rows, out_w, out_h, up, sr, overlap_sr=True, tag=""):
    """只驻留 VAE：逐张解码 + 超分 + 存盘。"""
    os.makedirs(OUT, exist_ok=True)
    from diffusers.models import AutoencoderKLQwenImage21
    t0 = time.perf_counter()
    vae = AutoencoderKLQwenImage21.from_pretrained(FULL, subfolder="vae",
                                                   torch_dtype=torch.float32,
                                                   local_files_only=True).eval()
    tiling = max(out_w, out_h) >= 1024
    if tiling:
        vae.enable_tiling()
    print(f"  [B] VAE 载入 {time.perf_counter()-t0:.1f}s | tiling={tiling} | RSS {rss():.2f} GB", flush=True)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    res = []
    q = queue.Queue()
    done = []

    def worker():
        from sr_models import upscale_array
        while True:
            item = q.get()
            if item is None:
                return
            arr, path, idx = item
            t1 = time.perf_counter()
            up_res = upscale_array(arr, name=sr, out_w=out_w, out_h=out_h) if (sr and up != "none") else (arr, 1)
            arr2 = up_res[0]
            Image.fromarray(arr2).save(path)
            done.append(dict(idx=idx, path=path, sr_s=round(time.perf_counter() - t1, 1),
                             shape=list(arr2.shape)))

    th = None
    if overlap_sr:
        th = threading.Thread(target=worker, daemon=True)
        th.start()

    for r in rows:
        lat = torch.from_numpy(np.load(r["latent"]))
        ch, h_lat, w_lat = r["shape"]
        t1 = time.perf_counter()
        z = lat.reshape(1, h_lat * w_lat, ch).transpose(1, 2).reshape(1, ch, 1, h_lat, w_lat)
        img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0][:, :3]
        arr = ((img.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).permute(0, 2, 3, 1)[0].numpy()
        vae_s = time.perf_counter() - t1
        path = os.path.join(OUT, f"batch2_{tag}_{r['idx']}.png")
        if overlap_sr:
            q.put((arr, path, r["idx"]))
            print(f"  [B] {r['idx']} 解码 {vae_s:.1f}s → 超分投递后台 | RSS {rss():.2f} GB", flush=True)
        else:
            from sr_models import upscale_array
            t2 = time.perf_counter()
            up_res = upscale_array(arr, name=sr, out_w=out_w, out_h=out_h) if (sr and up != "none") else (arr, 1)
            arr2 = up_res[0]
            Image.fromarray(arr2).save(path)
            done.append(dict(idx=r["idx"], path=path, sr_s=round(time.perf_counter() - t2, 1),
                             shape=list(arr2.shape)))
            print(f"  [B] {r['idx']} 解码 {vae_s:.1f}s + 超分 {time.perf_counter()-t2:.1f}s", flush=True)
        res.append(dict(idx=r["idx"], vae_s=round(vae_s, 1)))
    if overlap_sr:
        q.put(None)
        th.join()
    return dict(decode=res, outputs=sorted(done, key=lambda d: d["idx"]),
                stage_s=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--preset", default="1080p", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sigmas", default="1.0,0.75,0.5,0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--igpu", default="none", choices=["none", "fp16", "fp32"])
    ap.add_argument("--sr", default=DEFAULT_SR)
    ap.add_argument("--overlap-sr", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--size", type=int, default=0, help="覆盖预设的生成尺寸（正方形，测试用）")
    a = ap.parse_args()
    sigmas = [float(x) for x in a.sigmas.split(",")]
    gw, gh, up, ow, oh = PRESETS[a.preset]
    if a.size:
        gw = gh = a.size
        ow = oh = a.size * 2 if up != "none" else a.size
    t_all = time.perf_counter()

    print(f"  == 阶段 A：降噪 ×{len(a.prompts)}（{gw}x{gh}，{a.steps} 步，iGPU={a.igpu}）", flush=True)
    A = stage_a(a.prompts, gw, gh, a.steps, sigmas, a.seed, a.igpu, threads=a.threads)
    print(f"  == 阶段 B：解码 + 超分（{ow}x{oh}）", flush=True)
    B = stage_b(A["rows"], ow, oh, up, a.sr, overlap_sr=bool(a.overlap_sr), tag=a.tag)
    total = time.perf_counter() - t_all
    out = dict(preset=a.preset, n=len(a.prompts), igpu=a.igpu, steps=a.steps,
               gen=[gw, gh], out=[ow, oh], stage_a_s=A["stage_s"], stage_b_s=B["stage_s"],
               decode=A["rows"], outputs=B["outputs"], total_s=round(total, 1))
    with open(os.path.join(OUT, f"report_{a.preset}_{a.igpu}{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
