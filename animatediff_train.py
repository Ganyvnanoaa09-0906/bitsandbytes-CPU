# -*- coding: utf-8 -*-
"""AnimateDiff 5D training helpers (CPU, fp32) — imported by train_diffusion.py --video_selftest."""
from __future__ import annotations
import os, sys, time, json, threading, gc
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import psutil
import bitsandbytes as bnb
from diffusers import UNet2DConditionModel, MotionAdapter
from diffusers.models import UNetMotionModel
from diffusion_backends import apply_method

def resolve_video_model(spec: str):
    """spec: '<unet_dir>,<motion_adapter_dir>' or a directory containing unet/ + motion_adapter/."""
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        base = parts[0]
        return os.path.join(base, "unet"), os.path.join(base, "motion_adapter")
    raise ValueError("--video_model must be '<unet_dir>,<adapter_dir>' or a dir with unet/ + motion_adapter/")

def load_animatediff_model(unet_dir: str, adapter_dir: str):
    u = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=torch.float32, local_files_only=True)
    a = MotionAdapter.from_pretrained(adapter_dir, torch_dtype=torch.float32, local_files_only=True)
    m = UNetMotionModel.from_unet2d(u, a).to(torch.float32)
    del u, a
    gc.collect()
    return m
def _rss_gb():
    return psutil.Process().memory_info().rss / 1e9

class _Peak:
    def __init__(self):
        self.peak = _rss_gb(); self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, _rss_gb()); time.sleep(0.01)
    def start(self): self.t.start()
    def done(self):
        self.stop.set(); self.t.join(timeout=0.2); return self.peak

def run_animatediff_selftest(args) -> dict:
    """Minimal 5D training smoke on a real AnimateDiff model: Conv2d + Temporal Attention."""
    unet_dir, adapter_dir = resolve_video_model(args.video_model)
    torch.manual_seed(args.seed)
    model = load_animatediff_model(unet_dir, adapter_dir)
    model.train()
    model.enable_gradient_checkpointing()
    print(f"[video] loaded unet={unet_dir} adapter={adapter_dir} rss={_rss_gb():.2f}GB", flush=True)

    setup = apply_method("animatediff_lora", video_transformer=model, rank=args.rank,
                         alpha=args.alpha, dropout=args.lora_dropout,
                         target_modules=[x.strip() for x in args.targets.split(",") if x.strip()],
                         lora_scope=getattr(args, "lora_scope", "all"))
    model = setup.extra.get("transformer", model)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
    print(f"[video] trainable={n_train/1e6:.3f}M rss={_rss_gb():.2f}GB", flush=True)

    B, T, S = max(1, args.batch), max(1, args.video_frames), max(8, args.video_size)
    H = W = S // 8
    latents = torch.randn(B, 4, T, H, W)
    target = torch.randn_like(latents)
    e = torch.randn(B, 77, getattr(model.config, "cross_attention_dim", 768)).repeat(T, 1, 1)

    peak = _Peak(); peak.start()
    losses, times = [], []
    steps = max(1, min(10, args.steps))
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        noisy = latents + 0.1 * torch.randn_like(latents)
        ts = torch.randint(0, 1000, (B,))
        pred = model(noisy, ts, encoder_hidden_states=e).sample
        loss = F.mse_loss(pred.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        losses.append(float(loss.detach())); times.append(dt)
        print(f"[video step {step}] loss={losses[-1]:.6f} time={dt:.3f}s rss={_rss_gb():.3f}GB", flush=True)
    peak_rss = peak.done()
    return dict(kind="animatediff_selftest", unet_dir=unet_dir, adapter_dir=adapter_dir,
                trainable_params=int(n_train), input_shape=[B, 4, T, H, W], steps=steps,
                losses=losses, times_s=times, mean_step_s=sum(times)/len(times),
                peak_rss_gb=float(peak_rss))

class PairDataset(Dataset):
    """读取 clean_pipeline 产物: metadata.jsonl 中的 (image, pose) 对。"""
    def __init__(self, meta_path: str, res: int = 512):
        self.records = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("keep", True) and r.get("image"):
                    self.records.append(r)
        self.res = res

    def __len__(self):
        return len(self.records)

    def _load(self, path, res):
        im = Image.open(path).convert("RGB").resize((res, res), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(a).permute(2, 0, 1).contiguous()

    def __getitem__(self, i):
        r = self.records[i]
        img = self._load(r["image"], self.res)
        pose_path = r.get("pose")
        pose = self._load(pose_path, self.res) if pose_path and os.path.exists(pose_path) else torch.zeros_like(img)
        return dict(image=img, pose=pose, prompt=r.get("prompt", ""))
def run_animatediff_real_smoke(args) -> dict:
    """真实图像 (image, pose) 对 -> VAE latent -> AnimateDiff LoRA 训练 10 步冒烟。"""
    unet_dir, adapter_dir = resolve_video_model(args.video_model)
    torch.manual_seed(args.seed)
    model = load_animatediff_model(unet_dir, adapter_dir)
    model.train(); model.enable_gradient_checkpointing()

    vae = None
    if getattr(args, "vae_model", "") and os.path.isdir(args.vae_model):
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float32, local_files_only=True)
        vae.eval(); vae.requires_grad_(False)
        print(f"[video-real] vae={args.vae_model} rss={_rss_gb():.2f}GB", flush=True)

    meta = os.path.join(args.data, "metadata.jsonl") if args.data else ""
    ds = PairDataset(meta, args.video_size) if meta and os.path.exists(meta) else None
    print(f"[video-real] samples={len(ds) if ds is not None else 0} res={args.video_size} T={args.video_frames}", flush=True)

    setup = apply_method("animatediff_lora", video_transformer=model, rank=args.rank,
                         alpha=args.alpha, dropout=args.lora_dropout,
                         target_modules=[x.strip() for x in args.targets.split(",") if x.strip()],
                         lora_scope=getattr(args, "lora_scope", "all"))
    model = setup.extra.get("transformer", model)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
    print(f"[video-real] trainable={n_train/1e6:.3f}M rss={_rss_gb():.2f}GB", flush=True)
    B = max(1, args.batch)
    T = max(1, args.video_frames)
    S = max(16, args.video_size)
    H = W = S // 8
    cross_dim = getattr(getattr(model, "config", None), "cross_attention_dim", 768) or 768

    loader = None
    if ds is not None:
        loader = DataLoader(ds, batch_size=B, shuffle=True, drop_last=False)
        it = iter(loader)
    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    peak = _Peak(); peak.start()
    losses, times = [], []
    steps = max(1, min(10, args.steps))
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        if loader is not None:
            batch = next_batch()
            images = batch["image"].float()
            pose = batch["pose"].float()
            if vae is not None:
                with torch.no_grad():
                    z = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            else:
                z = torch.randn(images.shape[0], 4, H, W)
        else:
            z = torch.randn(B, 4, H, W)
            pose = torch.zeros(B, 3, S, S)
        latents = z.unsqueeze(2).repeat(1, 1, T, 1, 1).contiguous()
        target = torch.randn_like(latents)
        e = torch.randn(latents.shape[0], 77, cross_dim).repeat(T, 1, 1)
        noisy = latents + 0.1 * torch.randn_like(latents)
        ts = torch.randint(0, 1000, (latents.shape[0],))
        pred = model(noisy, ts, encoder_hidden_states=e).sample
        loss = F.mse_loss(pred.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        losses.append(float(loss.detach())); times.append(dt)
        print(f"[video-real step {step}] loss={losses[-1]:.6f} time={dt:.3f}s rss={_rss_gb():.3f}GB pose_mean={float(pose.mean()):.3f}", flush=True)
    peak_rss = peak.done()
    return dict(kind="animatediff_real_smoke", samples=len(ds) if ds is not None else 0,
                vae=bool(vae is not None), trainable_params=int(n_train),
                input_shape=[B, 4, T, H, W], steps=steps, losses=losses, times_s=times,
                mean_step_s=sum(times) / len(times), peak_rss_gb=float(peak_rss))
