# -*- coding: utf-8 -*-
"""qwen_batch.py — 一条命令批量出图（两阶段 + 断点续跑 + 内存保护）。

设计（吸收 A.16.24.16–.18 的教训）：
  阶段 A：transformer 驻留，逐张降噪，latent 落盘（严格 no_grad；已存在则跳过）
  阶段 B：释放 transformer，只驻留 VAE，逐张「解码 → 超分 → 存盘」**串行**（并发会冻机）
  内存保护：每张开始前检查可用内存，低于阈值先 gc + 等待；VAE 解码与超分不同线程并发。
  断点续跑：latent / 输出已存在即跳过，重跑同一条命令不会重复计算。

用法：
  python qwen_batch.py --preset 1080p --prompts "a cat" "a fox" "a lake"
  python qwen_batch.py --preset 1080p --prompts-file prompts.txt --tag v1
  python qwen_batch.py --preset 1080p --stage a      # 只跑生成
  python qwen_batch.py --preset 1080p --stage b      # 只跑解码/超分
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

from qwen4bit_loader import load_transformer
from qwen_lightning_bench import LORA, LoRAInjector, calculate_shift
from qwen_img import FULL, PRESETS, VAE_SCALE, prompt_embeds
from sr_models import DEFAULT as DEFAULT_SR

ROOT = r"D:\work\research\bench_results\qwen_batch"
MEM_FLOOR_GB = 1.5      # 可用内存低于此值就先 gc/等待


def rss():
    return psutil.Process().memory_info().rss / 1e9


def avail():
    return psutil.virtual_memory().available / 1e9


def mem_guard(tag=""):
    """内存保护：不足阈值时 gc，仍不足则等待（最多 60 s）。"""
    if avail() >= MEM_FLOOR_GB:
        return
    gc.collect()
    t0 = time.time()
    while avail() < MEM_FLOOR_GB and time.time() - t0 < 60:
        print(f"  [mem] 可用 {avail():.2f} GB < {MEM_FLOOR_GB} GB，等待… {tag}", flush=True)
        time.sleep(3)
        gc.collect()


def load_meta(outdir):
    p = os.path.join(outdir, "meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_meta(outdir, meta):
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def stage_a(prompts, gen_w, gen_h, steps, sigmas, seed, threads, outdir, igpu="none"):
    latdir = os.path.join(outdir, "latents")
    os.makedirs(latdir, exist_ok=True)
    torch.set_num_threads(threads)
    rows = []
    todo = []
    for i, p in enumerate(prompts):
        lp = os.path.join(latdir, f"lat_{i}.npy")
        if os.path.exists(lp):
            rows.append(dict(idx=i, prompt=p, latent=lp, skipped=True))
        else:
            todo.append((i, p, lp))
    if not todo:
        print(f"  [A] 全部 latent 已存在（{len(rows)} 张），跳过生成", flush=True)
        return rows

    t0 = time.time()
    tr = load_transformer()
    if igpu != "none":
        from qwen4bit_igpu_linear_fp32 import to_igpu_f32
        to_igpu_f32(tr, mode_fn=(lambda n: "fp16") if igpu == "fp16" else None)
    inj = LoRAInjector(tr, LORA, scale=1.0)
    load_s = time.time() - t0
    h_lat, w_lat = gen_h // VAE_SCALE, gen_w // VAE_SCALE
    with open(os.path.join(FULL, "vae", "config.json"), encoding="utf-8") as f:
        ch = int(json.load(f)["z_dim"])

    from diffusers import FlowMatchEulerDiscreteScheduler
    for i, p, lp in todo:
        mem_guard(f"A{i}")
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
        t1 = time.time()
        with torch.no_grad():          # 必须！否则每步保留计算图 → 内存暴涨
            for t in sched.timesteps:
                pred = tr(hidden_states=lat, timestep=(t.expand(1).to(lat.dtype) / 1000.0),
                          encoder_hidden_states=pe, encoder_hidden_states_mask=pm,
                          img_shapes=shapes, img_mask=img_mask, return_dict=False)[0][:, -seq:]
                lat = sched.step(pred, t, lat, return_dict=False)[0]
        dn = time.time() - t1
        np.save(lp, lat.detach().numpy().astype(np.float32))
        rows.append(dict(idx=i, prompt=p, latent=lp, denoise_s=round(dn, 1), shape=[ch, h_lat, w_lat]))
        print(f"  [A] {i} 降噪 {dn:.1f}s → {os.path.basename(lp)} | RSS {rss():.2f} GB "
              f"| 可用 {avail():.2f} GB", flush=True)
        del lat, embeds
        gc.collect()

    inj.remove()
    del tr, inj
    gc.collect()
    print(f"  [A] 生成完成 总 {time.time()-t0:.1f}s（含加载 {load_s:.1f}s）| RSS {rss():.2f} GB", flush=True)
    return rows


def stage_b(rows, gen_w, gen_h, out_w, out_h, up, sr, threads, outdir, tag=""):
    if not sr or up == "none":
        print("  [B] 无需解码/超分", flush=True)
        return []
    os.makedirs(outdir, exist_ok=True)
    todo = []
    done = []
    for r in rows:
        out = os.path.join(outdir, f"{tag or 'img'}_{r['idx']}.png")
        if os.path.exists(out):
            done.append(dict(idx=r["idx"], path=out, skipped=True))
        else:
            todo.append((r, out))
    if not todo:
        print(f"  [B] 全部输出已存在（{len(done)} 张），跳过解码", flush=True)
        return done

    from diffusers.models import AutoencoderKLQwenImage21
    t0 = time.time()
    vae = AutoencoderKLQwenImage21.from_pretrained(FULL, subfolder="vae",
                                                   torch_dtype=torch.float32,
                                                   local_files_only=True).eval()
    if max(out_w, out_h) >= 1024 or max(gen_w, gen_h) >= 1024:
        vae.enable_tiling()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    print(f"  [B] VAE 载入 {time.time()-t0:.1f}s | tiling | RSS {rss():.2f} GB", flush=True)

    h_lat, w_lat = gen_h // 16, gen_w // 16
    from sr_models import upscale_array
    for r, out in todo:
        mem_guard(f"B{r['idx']}")
        lat = torch.from_numpy(np.load(r["latent"]))
        seq_dim, ch = lat.shape[1], lat.shape[-1]
        t1 = time.time()
        z = lat.reshape(1, seq_dim, ch).transpose(1, 2).reshape(1, ch, 1, h_lat, w_lat)
        with torch.no_grad():           # 解码也不要计算图
            img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0][:, :3]
        arr = ((img.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).permute(0, 2, 3, 1)[0].numpy()
        vae_s = time.time() - t1
        del img, z, lat
        gc.collect()
        mem_guard(f"B{r['idx']}-sr")
        t2 = time.time()
        arr2 = upscale_array(arr, name=sr, out_w=out_w, out_h=out_h)[0]
        sr_s = time.time() - t2
        from PIL import Image
        Image.fromarray(arr2).save(out)
        print(f"  [B] {r['idx']} 解码 {vae_s:.1f}s + 超分 {sr_s:.1f}s → {os.path.basename(out)} "
              f"{list(arr2.shape)} | RSS {rss():.2f} GB | 可用 {avail():.2f} GB", flush=True)
        done.append(dict(idx=r["idx"], path=out, vae_s=round(vae_s, 1), sr_s=round(sr_s, 1),
                         shape=list(arr2.shape)))
        del arr, arr2
        gc.collect()
    return sorted(done, key=lambda d: d["idx"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="*", default=[])
    ap.add_argument("--prompts-file", default="")
    ap.add_argument("--preset", default="1080p", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sigmas", default="1.0,0.75,0.5,0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sr", default=DEFAULT_SR)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--igpu", default="none", choices=["none", "fp16", "fp32"])
    ap.add_argument("--stage", default="ab", choices=["a", "b", "ab"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()

    prompts = list(a.prompts)
    if a.prompts_file:
        with open(a.prompts_file, encoding="utf-8") as f:
            prompts += [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        raise SystemExit("需要 --prompts 或 --prompts-file")
    sigmas = [float(x) for x in a.sigmas.split(",")]
    gw, gh, up, ow, oh = PRESETS[a.preset]
    outdir = a.outdir or os.path.join(ROOT, f"{a.preset}{('_' + a.tag) if a.tag else ''}")
    os.makedirs(outdir, exist_ok=True)
    meta = load_meta(outdir)
    meta.update(dict(preset=a.preset, gen=[gw, gh], out=[ow, oh], steps=a.steps,
                     sigmas=sigmas, seed=a.seed, sr=a.sr, igpu=a.igpu, prompts=prompts))
    save_meta(outdir, meta)

    t_all = time.time()
    report = dict(outdir=outdir, preset=a.preset, n=len(prompts), gen=[gw, gh], out=[ow, oh])
    rows = []
    if "a" in a.stage:
        print(f"  == 阶段 A：生成 {len(prompts)} 张（{gw}x{gh}，{a.steps} 步）", flush=True)
        rows = stage_a(prompts, gw, gh, a.steps, sigmas, a.seed, a.threads, outdir, a.igpu)
        report["stage_a"] = rows
    else:
        latdir = os.path.join(outdir, "latents")
        for i, p in enumerate(prompts):
            lp = os.path.join(latdir, f"lat_{i}.npy")
            if os.path.exists(lp):
                rows.append(dict(idx=i, prompt=p, latent=lp))
    if "b" in a.stage:
        print(f"  == 阶段 B：解码 + 超分（{ow}x{oh}）", flush=True)
        report["stage_b"] = stage_b(rows, gw, gh, ow, oh, up, a.sr, a.threads, outdir, a.tag)

    report["total_s"] = round(time.time() - t_all, 1)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps({k: report[k] for k in ("outdir", "n", "total_s")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
