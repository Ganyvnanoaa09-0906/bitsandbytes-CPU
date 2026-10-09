"""train_animatediff_lora.py -- a real AnimateDiff LoRA trainer for this fork.

Why a new file: train_diffusion.py:17 says the video families have injection
wired but the training loop is not yet connected, and both existing video entry
points are smoke tests -- run_animatediff_selftest and run_animatediff_real_smoke
cap themselves at 10 steps (steps = max(1, min(10, args.steps))) and the "real"
one deliberately trains against torch.randn_like(latents) with random text
embeddings. They verify shapes, speed and memory, which is what let the chain be
declared working; they do not train anything.

This does the real thing:

  * latents come from the SD1.5 VAE (not randn)
  * the noise level and the target come from the SD1.5 scheduler, so the
    objective is the usual epsilon-prediction one
  * the text conditioning comes from the SD1.5 CLIP text encoder
  * encoder_hidden_states is expanded to (B*T, 77, D). Per VIDEO_GUIDE section 3
    UNetMotionModel reshapes (B,C,F,H,W) to (B*F,C,H,W) internally, so a
    (B,77,D) context raises a size mismatch. This is the trap that section exists
    for.
  * checkpoints, resume, and the fork's own optimisers (AdamW8bit by default,
    AdamW4bit optional)

Kept deliberately close to the smoke test's plumbing, which is already verified:
same loader, same LoRA injection through apply_method, same loop over steps.

Data: a metadata.jsonl where each line has at least {"image": path}. "pose" and
"prompt" are optional -- PairDataset zero-fills a missing pose and the caption
defaults to "".

Usage:
  python train_animatediff_lora.py \
    --data <dir with metadata.jsonl> --sd <dir with unet/vae/text_encoder/tokenizer/scheduler> \
    --adapter <motion adapter dir> --out runs/adiff1 \
    --video_frames 4 --video_size 256 --batch 1 --steps 2000 \
    --lora_scope temporal --optim adamw8bit
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import bitsandbytes as bnb  # noqa: E402
from animatediff_train import (PairDataset, load_animatediff_model,  # noqa: E402
                               resolve_video_model, _rss_gb)


def _lora_state(model):
    """Only the adapter weights -- 336 tensors instead of 1589, 4.7MB instead of 5GB.

    Saving the full state_dict also saves the frozen SD1.5 UNet, which is what
    filled the disk. Nothing is lost: the base weights are reloaded from disk on
    resume, and the resume path uses strict=False.
    """
    sd = model.state_dict()
    # 'lora' alone: the adapter weights already sit under motion_modules.*,
    # and matching motion_modules as well pulled in the entire frozen motion
    # adapter, making checkpoints 1740 MB instead of ~5 MB.
    keep = {k: v for k, v in sd.items() if 'lora' in k.lower()}
    return keep if keep else sd


def build_optimizer(name, params, lr, wd):
    name = (name or 'adamw8bit').lower()
    if name == 'adamw8bit':
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd)
    if name == 'adamw4bit':
        from bitsandbytes.optim import AdamW4bit
        return AdamW4bit(params, lr=lr, weight_decay=wd)
    if name == 'adamw32bit':
        return bnb.optim.AdamW32bit(params, lr=lr, weight_decay=wd)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise SystemExit('unknown --optim %r' % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='dir containing metadata.jsonl')
    ap.add_argument('--sd', required=True, help='SD1.5 dir: unet/ vae/ text_encoder/ tokenizer/ scheduler/')
    ap.add_argument('--adapter', required=True, help='AnimateDiff motion adapter dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--video_frames', type=int, default=4)
    ap.add_argument('--video_size', type=int, default=256)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--alpha', type=int, default=8)
    ap.add_argument('--lora_dropout', type=float, default=0.0)
    ap.add_argument('--targets', default='to_q,to_v,to_k,to_out.0')
    ap.add_argument('--lora_scope', default='temporal', choices=['all', 'temporal', 'spatial'])
    ap.add_argument('--optim', default='adamw8bit',
                    choices=['adamw8bit', 'adamw4bit', 'adamw32bit', 'adamw'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt_every', type=int, default=200)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--threads', type=int, default=5)
    ap.add_argument('--time_limit_s', type=float, default=0.0)
    ap.add_argument('--trim_every', type=int, default=20,
                    help='gc.collect() every N steps. Measured on the VQGAN path: without '
                         'it RSS grows 7-17MB/step and the run dies in a few hundred '
                         'steps with 0xC0000005 in c10.dll.')
    ap.add_argument('--max_rss_gb', type=float, default=9.0,
                    help='save and exit cleanly above this RSS, so a driver can resume. 0=off.')
    ap.add_argument('--resume', default='')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    unet_dir, adapter_dir = resolve_video_model('%s/unet,%s' % (a.sd, a.adapter))
    model = load_animatediff_model(unet_dir, adapter_dir)
    model.train()
    model.enable_gradient_checkpointing()
    print('[train] unet=%s adapter=%s rss=%.2fGB' % (unet_dir, adapter_dir, _rss_gb()), flush=True)

    from diffusion_backends import apply_method
    setup = apply_method('animatediff_lora', video_transformer=model, rank=a.rank,
                         alpha=a.alpha, dropout=a.lora_dropout,
                         target_modules=[x.strip() for x in a.targets.split(',') if x.strip()],
                         lora_scope=a.lora_scope)
    model = setup.extra.get('transformer', model)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print('[train] lora_scope=%s trainable=%.3fM rss=%.2fGB'
          % (a.lora_scope, n_train / 1e6, _rss_gb()), flush=True)
    if n_train == 0:
        raise SystemExit('no trainable parameters -- lora_scope=%r matched nothing' % a.lora_scope)

    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    vae = AutoencoderKL.from_pretrained(os.path.join(a.sd, 'vae'),
                                        torch_dtype=torch.float32, local_files_only=True)
    vae.eval(); vae.requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(os.path.join(a.sd, 'scheduler'))
    tok = CLIPTokenizer.from_pretrained(os.path.join(a.sd, 'tokenizer'), local_files_only=True)
    te = CLIPTextModel.from_pretrained(os.path.join(a.sd, 'text_encoder'),
                                       torch_dtype=torch.float32, local_files_only=True)
    te.eval(); te.requires_grad_(False)
    print('[train] vae+scheduler+text_encoder ready rss=%.2fGB' % _rss_gb(), flush=True)

    ds = PairDataset(os.path.join(a.data, 'metadata.jsonl'), a.video_size)
    if len(ds) == 0:
        raise SystemExit('metadata.jsonl has no usable rows (need {"image": path})')
    print('[train] samples=%d res=%d T=%d batch=%d' % (len(ds), a.video_size, a.video_frames, a.batch),
          flush=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=False, num_workers=0)

    opt = build_optimizer(a.optim, trainable, a.lr, a.weight_decay)
    start = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        try:
            opt.load_state_dict(ck['opt'])
        except Exception as e:
            print('[train] optimizer state not restored: %s' % e, flush=True)
        start = int(ck.get('step', 0))
        print('[train] resumed from %s at step %d' % (a.resume, start), flush=True)

    H = W = a.video_size // 8
    T = a.video_frames
    B = a.batch
    it = iter(dl)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    t0 = time.time()
    losses = []
    for step in range(start + 1, a.steps + 1):
        ts0 = time.perf_counter()
        batch = next_batch()
        images = batch['image'].float()
        prompts = batch['prompt']

        with torch.no_grad():
            z = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            ti = tok(list(prompts), padding='max_length', max_length=77,
                     truncation=True, return_tensors='pt')
            emb = te(**ti).last_hidden_state            # (B, 77, D)

        latents = z.unsqueeze(2).repeat(1, 1, T, 1, 1).contiguous()
        noise = torch.randn_like(latents)
        t = torch.randint(0, sched.config.num_train_timesteps, (latents.shape[0],))
        noisy = sched.add_noise(latents, noise, t)

        # (B,77,D) -> (B*T,77,D): UNetMotionModel folds B and F together, so the
        # context must already be expanded. See VIDEO_GUIDE section 3.
        e = emb.repeat_interleave(T, dim=0).contiguous()

        pred = model(noisy, t, encoder_hidden_states=e).sample
        loss = F.mse_loss(pred.float(), noise.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.warmup and step <= a.warmup:
            lr = a.lr * step / a.warmup
            for g in opt.param_groups:
                g['lr'] = lr
        opt.step()

        dt = time.perf_counter() - ts0
        losses.append(float(loss.detach()))
        if step % a.log_every == 0 or step == start + 1:
            recent = sum(losses[-a.log_every:]) / len(losses[-a.log_every:])
            print('[adiff step %d/%d] loss=%.5f avg=%.5f %.2fs rss=%.2fGB'
                  % (step, a.steps, losses[-1], recent, dt, _rss_gb()), flush=True)

        if a.ckpt_every and step % a.ckpt_every == 0:
            d = dict(model=_lora_state(model), opt=opt.state_dict(), step=step,
                     lora_only=True,
                     config=dict(video_frames=T, video_size=a.video_size, rank=a.rank,
                                 alpha=a.alpha, lora_scope=a.lora_scope, optim=a.optim))
            torch.save(d, os.path.join(a.out, 'ckpt_%d.pt' % step))
            torch.save(d, os.path.join(a.out, 'ckpt_last.pt'))
            print('[adiff] saved ckpt_%d.pt' % step, flush=True)

        if a.trim_every and step % a.trim_every == 0:
            gc.collect()

        if a.max_rss_gb and _rss_gb() >= a.max_rss_gb:
            r = _rss_gb()
            print('[adiff] RSS %.2fGB >= --max_rss_gb %.2fGB, saving and exiting at step %d'
                  % (r, a.max_rss_gb, step), flush=True)
            torch.save(dict(model=_lora_state(model), opt=opt.state_dict(), step=step,
                            lora_only=True,
                            config=dict(video_frames=T, video_size=a.video_size, rank=a.rank,
                                        alpha=a.alpha, lora_scope=a.lora_scope,
                                        optim=a.optim)),
                       os.path.join(a.out, 'ckpt_last.pt'))
            return

        if a.time_limit_s and time.time() - t0 > a.time_limit_s:
            print('[adiff] time limit reached at step %d' % step, flush=True)
            break

    d = dict(model=_lora_state(model), opt=opt.state_dict(), step=step,
             lora_only=True,
             config=dict(video_frames=T, video_size=a.video_size, rank=a.rank,
                         alpha=a.alpha, lora_scope=a.lora_scope, optim=a.optim))
    torch.save(d, os.path.join(a.out, 'ckpt_last.pt'))
    n = len(losses)
    print('[adiff] done steps=%d first=%.5f last=%.5f mean_step=%.3fs peak_rss=%.2fGB'
          % (n, losses[0] if n else -1, losses[-1] if n else -1,
             (time.time() - t0) / max(1, n), _rss_gb()), flush=True)


if __name__ == '__main__':
    main()
