# -*- coding: utf-8 -*-
"""independent_chunks.py — 独立分块生成（E2E 管线版，2026-10-23 port）。

本文件由 E2E 管线副本 port 回仓库（feat/video-e2e，起点 824995d0）。
原始实现的行为与数学未改，只加接口/修 bug；所有改动见下方 E2E-PATCH 列表。
每个 chunk 从纯噪声跑完整 schedule、各自 seed、硬切拼接；per-shot 指标 + seam/intra MAD。

E2E-PATCH 列表（详见 PENDING_REPO_PORT.md）：
  [P1] LoRA 手动 merge 支持两种键前缀：
       - 现有 `base_model.model.<path>.lora_A/B[.default].weight`（PEFT 原始 state_dict）
       - `unet.<path>.lora_A/B.weight`（convert_lora_ckpt.py 的输出）
       原实现只认前者 ⇒ 用转换后的 safetensors 会 **0 层命中却返回成功**（静默）。现在
       `iter_lora_pairs()` 统一归一化，merge 后 **断言 applied>0**，否则 raise。
  [P2] 加 `--chunk-json`：把 per-shot 指标/耗时写进 `chunk_metrics.json`（编排器合并进统一
       metrics.json）；原 legacy `<tag>.json` 保留。
  [P3] 惰性 import：torch / safetensors 只在真正生成/merge 时导入，`--help` 不需要 torch。
  [P4] 输出目录支持 `--outdir`（原件已有），另加 `--frames-dir` 显式指定帧目录名，
       与管线约定 `frames/fNNN.png` 对齐（默认就是 frames/，保持原行为）。

用法:
  python independent_chunks_e2e.py --res 128 --chunk 8 --chunks 2 --steps 6 --outdir runs/x
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
import json as _json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

GUIDANCE = 7.5


def _reconfigure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def lapvar(a):
    a = a.astype(np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def fstats(a):
    a = a.astype(np.float32)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    return dict(lap=lapvar(a), white=float(100 * (a > 245).mean()),
                blown=float(((lum > 235) & (sat < 0.08)).mean() * 100),
                green=int(((g - np.maximum(r, b)) > 25).sum()),
                mean=float(a.mean()))


# E2E-PATCH[P1]: 键归一化（纯 dict 操作，不需要 torch）
_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight",
                  ".lora_A.default.weight", ".lora_B.default.weight")


def normalize_lora_key(k):
    """把 LoRA 键归一化成 `<路径>.lora_A.weight` / `<路径>.lora_B.weight` 或返回 None。

    支持三种写法（E2E-PATCH[P1]）：
      base_model.model.down_blocks.0.attn1.to_q.lora_A.default.weight   # PEFT 原始
      base_model.model.down_blocks.0.attn1.to_q.lora_A.weight
      unet.down_blocks.0.attn1.to_q.lora_A.weight                       # convert_lora_ckpt.py 输出
      down_blocks.0.attn1.to_q.lora_A.weight                            # 无前缀（兼容）
    """
    rest = k
    for pref in ("base_model.model.", "unet."):
        if rest.startswith(pref):
            rest = rest[len(pref):]
            break
    rest = rest.replace(".lora_A.default.weight", ".lora_A.weight")
    rest = rest.replace(".lora_B.default.weight", ".lora_B.weight")
    if rest.endswith(".lora_A.weight") or rest.endswith(".lora_B.weight"):
        return rest
    return None


def iter_lora_pairs(sd):
    """从 LoRA state_dict 提取 {module_path: {'A': tensor, 'B': tensor}}（不依赖 torch）。"""
    pairs = {}
    for k, v in sd.items():
        nk = normalize_lora_key(k)
        if nk is None:
            continue
        if nk.endswith(".lora_A.weight"):
            pairs.setdefault(nk[:-len(".lora_A.weight")], {})["A"] = v
        else:
            pairs.setdefault(nk[:-len(".lora_B.weight")], {})["B"] = v
    return {p: ab for p, ab in pairs.items() if "A" in ab and "B" in ab}


def main():
    _reconfigure_stdout()
    ap = argparse.ArgumentParser(description="independent chunk generation (E2E copy)")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--neg", default=None)
    ap.add_argument("--tag", default="shots")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--frames-dir", default="frames", help="帧目录名（管线约定 frames/）")
    ap.add_argument("--chunk-json", default="", help="per-shot 指标落盘路径")
    ap.add_argument("--lora", nargs="*", default=[],
                    help="zero or more LoRA dirs/files, merged into the UNet in order")
    ap.add_argument("--lora-scale", nargs="*", type=float, default=[],
                    help="scale per LoRA, positionally matched (default 0.8 each)")
    ap.add_argument("--lora-method", default="manual", choices=["auto", "peft", "manual"])
    # E2E-PATCH[P8]: adapter 版本必须与训练一致（v1-5-2）。anime_adiff.build 默认是 v1-5，
    #   而 v1-5 没有 mid_block 的时序层 ⇒ 自训 LoRA 的 8 个 mid_block 模块会 merge 不进去
    #   （实测 160/168）。默认改成 v1-5-2 后 168/168 全部命中。
    ap.add_argument("--adapter", default=r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2")
    a = ap.parse_args()

    # E2E-PATCH[P3]: 惰性 import —— 只有真正生成才需要 torch/diffusers
    import torch
    from PIL import Image
    from scripts.lib.anime_adiff import build, ANIME_PROMPT, ANIME_NEG, OUT_DIR, VAE_DIR

    prompt = a.prompt if a.prompt is not None else ANIME_PROMPT
    neg = a.neg if a.neg is not None else ANIME_NEG

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    outdir = a.outdir or os.path.join(OUT_DIR, f"{a.tag}_{a.res}_{a.chunk}f{a.chunks}n")
    os.makedirs(outdir, exist_ok=True)

    print(f"[cfg] res={a.res} chunk={a.chunk} chunks={a.chunks} steps={a.steps} "
          f"mode=INDEPENDENT (full schedule per chunk, hard cuts)")
    print(f"[cfg] vae={VAE_DIR} adapter={a.adapter} outdir={outdir}")

    pipe = build("dpm", adapter=a.adapter)

    _done = []

    def merge_lora(path, lora_scale):
        """W += lora_scale * (alpha/r) * (B @ A) for every adapter pair found in `path`."""
        ok = False
        if a.lora_method in ("auto", "peft"):
            try:
                pipe.load_lora_weights(path, adapter_name=f"a{len(_done)}")
                pipe.set_adapters([f"a{len(_done)}"], adapter_weights=[lora_scale])
                print(f"[cfg] lora loaded via peft: {path} (scale={lora_scale})")
                ok = True
            except Exception as e:
                print(f"[info] peft path unavailable ({str(e)[:80]})")
                if a.lora_method == "peft":
                    raise
        if not ok:
            import json as _json
            from safetensors.torch import load_file as _lf
            p = path
            alpha = r = None
            if os.path.isdir(p):
                sd = _lf(os.path.join(p, "adapter_model.safetensors"))
                cfgp = os.path.join(p, "adapter_config.json")
                if os.path.exists(cfgp):
                    c = _json.load(open(cfgp, encoding="utf-8"))
                    alpha, r = c.get("lora_alpha"), c.get("r")
            else:
                sd = _lf(p)
            scale = lora_scale * ((alpha / r) if (alpha and r) else 1.0)
            mods = dict(pipe.unet.named_modules())
            pairs = iter_lora_pairs(sd)          # E2E-PATCH[P1]
            applied = 0
            for path_, ab in pairs.items():
                if path_ not in mods:
                    continue
                mod = mods[path_]
                if not hasattr(mod, "weight"):
                    continue
                delta = (ab["B"].float() @ ab["A"].float()) * scale
                with torch.no_grad():
                    mod.weight.add_(delta.to(mod.weight.dtype))
                applied += 1
            print(f"[cfg] lora merged manually: {applied}/{len(pairs)} layers, scale={scale:.3f} "
                  f"(alpha={alpha}, r={r}) <- {path}")
            # E2E-PATCH[P1]: 0 层命中必须响亮失败（原实现会静默"成功"）
            if applied == 0:
                raise SystemExit(
                    "[err] LoRA merge 命中 0 层！检查键前缀（base_model.model. / unet.）"
                    f"与模块名是否匹配；pairs={len(pairs)} 例: {list(pairs)[:3]}")
            return applied
        return -1

    if a.lora:
        scales = list(a.lora_scale)
        while len(scales) < len(a.lora):
            scales.append(scales[-1] if scales else 0.8)
        for lp, ls in zip(a.lora, scales):
            merge_lora(lp, ls)
            _done.append(lp)
        print(f"[cfg] {len(_done)} LoRA(s) merged; total delta is the SUM of all adapters")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(prompt, "cpu", 1, True, negative_prompt=neg)
    # E2E-PATCH[P6]: 必须按帧展开！UNetMotionModel 内部把 (B,C,F,H,W) reshape 成 (B*F,C,H,W)，
    #   所以 cross-attention 的 context 也要是 (B*F,77,768)。原脚本直接给 (2,77,768)，
    #   在 diffusers 0.40 下会报 `size of tensor a (2048) must match tensor b (256)`
    #   （比值=帧数，正是 VIDEO_GUIDE §3 记的坑）。B=2（CFG）× F=chunk ⇒ repeat_interleave(chunk)。
    embeds = torch.cat([ne, pe]).repeat_interleave(a.chunk, dim=0).contiguous()

    lat_h = a.res // pipe.vae_scale_factor
    sched = pipe.scheduler
    per_chunk = []
    all_frames = []
    t_all = time.time()

    for ci in range(a.chunks):
        sched.set_timesteps(a.steps)
        sched._step_index = 0
        sched.model_outputs = [None] * sched.config.solver_order
        sched._lower_order_nums = 0
        ts_run = list(sched.timesteps)

        gen = torch.Generator().manual_seed(a.seed + ci * 1000)
        lat = torch.randn((1, pipe.unet.config.in_channels, a.chunk, lat_h, lat_h),
                          generator=gen, dtype=torch.float32)
        lat = lat * sched.init_noise_sigma
        t0 = time.time()
        with torch.no_grad():
            for t in ts_run:
                mi = torch.cat([lat] * 2)
                mi = sched.scale_model_input(mi, t)
                npr = pipe.unet(mi, t, encoder_hidden_states=embeds).sample
                nu, nt = npr.chunk(2)
                npr = nu + GUIDANCE * (nt - nu)
                lat = sched.step(npr, t, lat,
                                 **pipe.prepare_extra_step_kwargs(gen, 0.0)).prev_sample

        with torch.no_grad():
            b, c, T, h, w = lat.shape
            flat = (lat / pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4
                                                                  ).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            arr = (((dec / 2 + 0.5).clamp(0, 1)).reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                   .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype(np.uint8)

        st = [fstats(arr[i]) for i in range(len(arr))]
        secs = time.time() - t0
        rec = dict(chunk=ci, seed=a.seed + ci * 1000, steps=len(ts_run),
                   seconds=secs, s_per_step=secs / max(1, len(ts_run)),
                   lap=float(np.mean([s["lap"] for s in st])),
                   white=float(np.mean([s["white"] for s in st])),
                   blown=float(np.mean([s["blown"] for s in st])),
                   green=int(np.sum([s["green"] for s in st])),
                   mean=float(np.mean([s["mean"] for s in st])))
        per_chunk.append(rec)
        print(f"  [shot {ci}] {rec['seconds']:.0f}s  lap={rec['lap']:>7.1f}  "
              f"white%={rec['white']:.2f}  green={rec['green']:>4}  mean={rec['mean']:.1f}",
              flush=True)
        all_frames.extend(list(arr))

    dt = time.time() - t_all

    fdir = os.path.join(outdir, a.frames_dir)
    os.makedirs(fdir, exist_ok=True)
    imgs = [Image.fromarray(f).convert("RGB") for f in all_frames]
    for i, im in enumerate(imgs):
        im.save(os.path.join(fdir, f"f{i:03d}.png"))
    gif = os.path.join(outdir, f"{a.tag}_{a.res}_{a.chunk}f{a.chunks}n_{a.steps}s.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=200, loop=0)
    strip = os.path.join(outdir, "sample_strip.png")
    try:
        sw = min(len(imgs), 16)
        sheet = Image.new("RGB", (imgs[0].width * sw, imgs[0].height))
        for i in range(sw):
            sheet.paste(imgs[i], (i * imgs[0].width, 0))
        sheet.save(strip)
    except Exception as e:
        print(f"[warn] strip 生成失败: {e}")

    laps = [r["lap"] for r in per_chunk]
    fr = [f.astype(np.float32) for f in all_frames]
    mads = [float(np.abs(fr[i + 1] - fr[i]).mean()) for i in range(len(fr) - 1)]
    seams = [mads[i] for i in range(len(mads)) if (i + 1) % a.chunk == 0]
    intra = [mads[i] for i in range(len(mads)) if (i + 1) % a.chunk != 0]

    print(f"\n{'='*72}")
    print(f"RESULT  independent shots  ({len(all_frames)} frames, {dt:.0f}s total)")
    print(f"  Laplacian per shot : " + " ".join(f"{v:.1f}" for v in laps))
    print(f"  detail spread      : {min(laps):.1f} .. {max(laps):.1f}  "
          f"(relay decayed 164 -> 72; independent should stay flat)")
    print(f"  white% per shot    : " + " ".join(f"{r['white']:.2f}" for r in per_chunk))
    print(f"  green px total     : {sum(r['green'] for r in per_chunk)}")
    print(f"  intra-shot MAD     : {np.mean(intra) if intra else float('nan'):.2f}")
    print(f"  SEAM MAD (cuts)    : {np.mean(seams) if seams else float('nan'):.2f}"
          f"   <- intentionally high = hard cut")
    print(f"  seconds per frame  : {dt/len(all_frames):.1f}")
    print(f"[save] {gif}")
    print(f"[save] {fdir}\\f000..f{len(imgs)-1:03d}.png")

    legacy = os.path.join(outdir, "metrics.json")
    with open(legacy, "w", encoding="utf-8") as fh:
        json.dump(dict(config=vars(a), vae=VAE_DIR, per_chunk=per_chunk,
                       lap=laps, intra_mad=float(np.mean(intra)) if intra else None,
                       seam_mad=float(np.mean(seams)) if seams else None,
                       seconds=dt, frames=len(all_frames)), fh, indent=2)

    # E2E-PATCH[P2]: per-shot 指标单独落盘，供编排器合并进统一 metrics.json
    chunk_json = a.chunk_json or os.path.join(outdir, "chunk_metrics.json")
    with open(chunk_json, "w", encoding="utf-8") as fh:
        json.dump(dict(mode="independent", chunk=a.chunk, overlap=0, chunks=a.chunks,
                       per_chunk=per_chunk, lap=laps,
                       intra_mad=float(np.mean(intra)) if intra else None,
                       seam_mad=float(np.mean(seams)) if seams else None,
                       seconds=dt, frames=len(all_frames), gif=gif, frames_dir=fdir),
                  fh, indent=2)
    print(f"[save] {chunk_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
