# -*- coding: utf-8 -*-
"""qwen_img_pipe.py — 批量出图 + VAE 解码流水线重叠（A.16.31 配方落地）。

思路：
  第 k 张做「降噪」（GEMM 在核显上，CPU 相对闲）时，让第 k-1 张的「VAE 解码 + 超分 + 存盘」
  在后台线程里跑 —— 正好对应 A.16.31 测到的「GPU 算 GEMM + CPU 干别的」重叠形态。

用法：
  python qwen_img_pipe.py --prompts "a cat" "a dog" --size 256 --steps 4 --igpu fp16 --overlap 1
  python qwen_img_pipe.py --... --overlap 0     # 对照组：全顺序
"""
from __future__ import annotations

import argparse
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
from qwen_img import FULL, VAE_SCALE, prompt_embeds
from sr_models import DEFAULT as DEFAULT_SR

OUT = r"D:\work\research\bench_results\qwen_pipe"


def rss():
    return psutil.Process().memory_info().rss / 1e9


class Pipeliner:
    def __init__(self, gen_w, gen_h, up, out_w, out_h, steps, sigmas, seed, igpu, sr, threads=6):
        self.gen_w, self.gen_h, self.up = gen_w, gen_h, up
        self.out_w, self.out_h = out_w, out_h
        self.size = max(gen_w, gen_h)
        self.steps, self.sigmas, self.seed = steps, sigmas, seed
        self.igpu, self.sr, self.threads = igpu, sr, threads
        torch.set_num_threads(threads)
        self.tr = load_transformer()
        if igpu != "none":
            from qwen4bit_igpu_linear_fp32 import to_igpu_f32
            to_igpu_f32(self.tr, mode_fn=(lambda n: "fp16") if igpu == "fp16" else None)
            print(f"  [iGPU] Linear 包装完成（{igpu}）", flush=True)
        self.inj = LoRAInjector(self.tr, LORA, scale=1.0)
        self._vae = None
        self.vae_tiling = max(self.gen_w, self.gen_h) >= 1024
        self.vae_load()   # 与 CLI 一致：先载入 VAE 再降噪
        self.q = queue.Queue()
        self.results = []
        with open(os.path.join(FULL, "vae", "config.json"), encoding="utf-8") as f:
            self.z_dim = int(json.load(f)["z_dim"])

    def vae_load(self):
        if self._vae is None:
            from diffusers.models import AutoencoderKLQwenImage21
            t0 = time.time()
            self._vae = AutoencoderKLQwenImage21.from_pretrained(FULL, subfolder="vae",
                                                                 torch_dtype=torch.float32,
                                                                 local_files_only=True).eval()
            if self.vae_tiling:
                self._vae.enable_tiling()
            print(f"  [VAE] 按需载入 {time.time()-t0:.1f}s | RSS {rss():.2f} GB", flush=True)
        return self._vae

    def vae_free(self):
        if self._vae is not None:
            self._vae = None
            import gc as _gc
            _gc.collect()
            print(f"  [VAE] 已释放 | RSS {rss():.2f} GB", flush=True)

    # ---------------- 降噪（GPU/CPU 主链路） ----------------
    def denoise(self, embeds, seed):
        from diffusers import FlowMatchEulerDiscreteScheduler
        w_px, h_px = self.gen_w, self.gen_h
        h_lat, w_lat = h_px // VAE_SCALE, w_px // VAE_SCALE
        ch = self.z_dim
        g = torch.Generator().manual_seed(seed)
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
        sched.set_timesteps(sigmas=self.sigmas, mu=calculate_shift(seq), device="cpu")
        ts = sched.timesteps
        t0 = time.perf_counter()
        with torch.no_grad():          # 关键：不加会保留计算图，RSS 会一路涨到 OOM/崩溃
            for i, t in enumerate(ts):
                pred = self.tr(hidden_states=lat, timestep=(t.expand(1).to(lat.dtype) / 1000.0),
                               encoder_hidden_states=pe, encoder_hidden_states_mask=pm,
                               img_shapes=shapes, img_mask=img_mask, return_dict=False)[0][:, -seq:]
                lat = sched.step(pred, t, lat, return_dict=False)[0]
        return lat, time.perf_counter() - t0, (h_lat, w_lat, ch)

    # ---------------- 解码 + 超分 + 存盘（CPU 侧，可后台） ----------------
    def decode(self, lat, meta):
        """VAE 解码留在主线程（内存峰值小，避免 OOM）。"""
        h_lat, w_lat, ch = meta
        t0 = time.perf_counter()
        vae = self.vae_load()
        z = lat.transpose(1, 2).reshape(1, ch, 1, h_lat, w_lat)
        mean = torch.tensor(vae.config.latents_mean).view(1, ch, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, ch, 1, 1, 1)
        img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0][:, :3]
        arr = ((img.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).permute(0, 2, 3, 1)[0].numpy()
        return arr, time.perf_counter() - t0

    def upscale_save(self, arr, path):
        t1 = time.perf_counter()
        from sr_models import upscale_array
        up = upscale_array(arr, name=self.sr, out_w=self.out_w, out_h=self.out_h) if (self.sr and self.up != "none") else (arr, 1)
        arr2 = up[0] if isinstance(up, tuple) else up
        sr_s = time.perf_counter() - t1
        Image.fromarray(arr2).save(path)
        return sr_s, arr2.shape

    def worker(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            arr, vae_s, path, idx = item
            sr_s, shape = self.upscale_save(arr, path)
            self.results.append(dict(idx=idx, path=path, vae_s=round(vae_s, 1), sr_s=round(sr_s, 1), shape=shape))

    def run(self, prompts, overlap=True, tag="", embeds=None):
        os.makedirs(OUT, exist_ok=True)
        paths = []
        th = None
        if overlap:
            th = threading.Thread(target=self.worker, daemon=True)
            th.start()
        t_all = time.perf_counter()
        per = []
        for i, p in enumerate(prompts):
            path = os.path.join(OUT, f"pipe_{tag}_{i}.png")
            paths.append(path)
            emb = embeds[i] if embeds else prompt_embeds(p)
            lat, dn_s, meta = self.denoise(emb, self.seed + i)
            arr, vae_s = self.decode(lat, meta)
            del lat
            self.vae_free()
            if overlap:
                self.q.put((arr, vae_s, path, i))
                print(f"  [{i}] 降噪 {dn_s:.1f}s + VAE {vae_s:.1f}s → 超分已投递后台 RSS {rss():.2f} GB", flush=True)
            else:
                sr_s, shape = self.upscale_save(arr, path)
                print(f"  [{i}] 降噪 {dn_s:.1f}s + VAE {vae_s:.1f}s + 超分 {sr_s:.1f}s", flush=True)
            per.append(dict(idx=i, denoise_s=round(dn_s, 1)))
        if overlap:
            self.q.put(None)
            th.join()
        total = time.perf_counter() - t_all
        self.inj.remove()
        return dict(overlap=overlap, n=len(prompts), size=self.size,
                    gen=[self.gen_w, self.gen_h], out=[self.out_w, self.out_h], steps=self.steps,
                    igpu=self.igpu, total_s=round(total, 1),
                    per_image=per, decodes=sorted(self.results, key=lambda r: r["idx"]),
                    outdir=OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--preset", default="", choices=["", "1024", "1080p", "4k", "native768"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sigmas", default="1.0,0.75,0.5,0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--igpu", default="fp16", choices=["none", "fp16", "fp32"])
    ap.add_argument("--sr", default=DEFAULT_SR)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    sigmas = [float(x) for x in a.sigmas.split(",")]
    from qwen_img import PRESETS
    if a.preset:
        gw, gh, up, ow, oh = PRESETS[a.preset]
    else:
        gw = gh = a.size; up = "x2"; ow = oh = a.size * 2
    t0 = time.perf_counter()
    embeds = [prompt_embeds(p) for p in a.prompts]     # 先把文本编码全部做完（8B 模型随后释放）
    import gc as _gc
    _gc.collect()
    print(f"  文本编码 {len(a.prompts)} 条完成 {time.perf_counter()-t0:.1f}s | RSS {rss():.2f} GB", flush=True)
    pl = Pipeliner(gw, gh, up, ow, oh, a.steps, sigmas, a.seed, a.igpu, a.sr, threads=a.threads)
    try:
        res = pl.run(a.prompts, overlap=bool(a.overlap), tag=a.tag or ("ov" if a.overlap else "seq"), embeds=embeds)
    finally:
        pass
    print("RESULT " + __import__("json").dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
