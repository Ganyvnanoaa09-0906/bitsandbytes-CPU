# -*- coding: utf-8 -*-
"""sample_latent_diffusion.py — DDIM 采样 + 用【同一个 VQ 解码器】出图，并给模式覆盖率。

关键：解码器与 AR 完全一致（C_mse 的 dec），所以出图口径可比。
额外给一个不需要 FID 的模式覆盖指标：
  把生成的潜变量和【真实验证集潜变量】各自分配到最近的码本条目，比较两个直方图。
  崩溃的模型会集中到少数码 ⇒ 直方图明显偏离真实分布。
  指标：总变差距离 TV = 0.5*sum|p_gen - p_real|，越接近 0 越好（另有 KL）。

用法:
  python sample_latent_diffusion.py --ckpt <.autopilot/diff_cmse/ckpt_last.pt> --vqvae ckpt_C_mse.pt --out grid.png
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt
from scripts.lib.train_latent_diffusion import build_unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--latents', default=r'D:\work\bitsandbytes-CPU\.autopilot\latents_cmse.pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--ddim-steps', type=int, default=50)
    ap.add_argument('--use-ema', action='store_true', default=True)
    ap.add_argument('--start-t', type=int, default=900,
                    help='从哪个 t 开始去噪。★ 关键：squaredcos_cap_v2 的 ac[999]=2.4e-9，'
                         'sqrt(ac)=5e-5 ⇒ DDIM 的 pred_x0=(x-√(1-ᾱ)ε)/√ᾱ 会把 ε 误差'
                         '放大 20291 倍，顶部时间步数值上无救（实测从 t=999 起动会发散成'
                         '彩虹噪点，std 1.0->3.8）。而 t=900 处 ac=0.0236，'
                         'Var(x)=ac·σ²+(1-ac)≈1.0 ⇒ N(0,I) 在 t=900 就已经是【正确的边缘分布】，'
                         '从 900 起跳既省掉病态区间又几乎无损（SD 的 scaled_linear 放大只有 37x）。')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--arch', default='unet', choices=['unet', 'looped'],
                    help='与 train_latent_diffusion.py 的 --arch 对应')
    ap.add_argument('--d', type=int, default=384)
    ap.add_argument('--n-unique', type=int, default=8)
    ap.add_argument('--loop-start', type=int, default=2)
    ap.add_argument('--loop-end', type=int, default=3)
    ap.add_argument('--n-loops', type=int, default=1)
    ap.add_argument('--loop-t-emb', action='store_true')
    ap.add_argument('--loop-damp', default='1/n')
    ap.add_argument('--infer-device', default='cpu', choices=['cpu', 'dml', 'auto'],
                    help='采样用哪个设备；实测核显端到端 DDIM 快 1.72x')
    # ---- 文本条件 + CFG（2026-09-23）----
    # 条件张量是【预计算好的字幕 CLIP 文本嵌入】，行号与 latents 一一对应。
    # 所以这里的用法是"用数据集中某一条的字幕作为条件"，而不是任意提示词
    # （任意提示词需要 CLIP 文本编码器，本机离线场景没有）。
    ap.add_argument('--cond-pt', default=None, help='条件张量 .pt，形状 (N, cond_dim)')
    ap.add_argument('--cond-index', type=int, default=None,
                    help='用第几行的条件；不给则每张图用不同行（从 --cond-start 起连续取）')
    ap.add_argument('--cond-start', type=int, default=0, help='--cond-index 未给时的起始行')
    ap.add_argument('--cfg-scale', type=float, default=1.0,
                    help='classifier-free guidance 强度；1.0=不做 CFG（只跑一次前向）')
    ap.add_argument('--prompt', default=None, nargs='+',
                    help='任意提示词（用 openai/clip-vit-base-patch32 编码，'
                         '已用原字幕回放验证与原 cond 余弦=1.0000）。'
                         '给一条则所有图共用；给多条则第 i 条对应第 i 张图。')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    # ★ 先加载 ckpt 再建调度器 —— 原版这里顺序反了：
    #   第 47 行就用 ck.get('beta_schedule')，而 ck 到第 63 行才 load，
    #   所以脚本一跑就 NameError。修：先 load。
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)

    ptype = str(ck.get('pred_type', 'epsilon'))
    from diffusers import DDIMScheduler
    bs = str(ck.get('beta_schedule', 'squaredcos_cap_v2'))
    b0 = float(ck.get('beta_start', 0.00085))
    b1 = float(ck.get('beta_end', 0.012))
    # ★ prediction_type 必须与训练时一致：v 目标的模型输出的是 v，
    #   调度器的 step() 会按这个字段换算回 x0/eps。用错就整条链都错。
    if bs == 'squaredcos_cap_v2':
        sch = DDIMScheduler(num_train_timesteps=1000, beta_schedule=bs,
                            clip_sample=True, clip_sample_range=4.0,
                            set_alpha_to_one=False, prediction_type=ptype)
    else:
        sch = DDIMScheduler(num_train_timesteps=1000, beta_schedule=bs,
                            beta_start=b0, beta_end=b1,
                            clip_sample=True, clip_sample_range=4.0,
                            set_alpha_to_one=False, prediction_type=ptype)
    print('[diff] 采样日程 = ckpt 训练日程: %s (beta %.5f..%.3f) acc=%.3e  预测目标=%s'
          % (bs, b0, b1, sch.alphas_cumprod[999], ptype), flush=True)
    sch.set_timesteps(a.ddim_steps)

    S = int(ck.get('sample_size', 16))
    Cin = int(ck.get('in_ch', 64))
    if a.arch == 'looped':
        from scripts.lib.looped_dit import LoopedDiT
        model = LoopedDiT(in_ch=Cin, latent_size=S, d=a.d, n_unique=a.n_unique,
                          loop_start=a.loop_start, loop_end=a.loop_end,
                          n_loops=a.n_loops, loop_t_emb=a.loop_t_emb,
                          loop_damp=a.loop_damp, cond_dim=int(ck.get('cond_dim', 0)))
    else:
        model = build_unet([int(x) for x in str(ck.get('ch', '128,256,256')).split(',')],
                           sample_size=S, in_ch=Cin)
    sd = ck.get('ema') if (a.use_ema and ck.get('ema')) else ck['model']
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.eval()
    print('[diff] ckpt step=%s  weights=%s  latent=%dx%d ch=%d  missing=%d unexpected=%d'
          % (ck.get('step'), 'ema' if (a.use_ema and ck.get('ema')) else 'model',
             S, S, Cin, len(miss), len(unexp)), flush=True)

    # ---- 推理设备：采样循环放核显（实测端到端 DDIM 20 NFE 快 1.72x）----
    from scripts.lib.infer_backend import InferBackend
    ibe = InferBackend(model, a.infer_device, verbose=True)

    g = torch.Generator().manual_seed(a.seed)
    x = torch.randn(a.n, Cin, S, S, generator=g)
    # ★ 丢掉顶部病态时间步（见 --start-t 的说明）
    all_ts = list(sch.timesteps)
    ts = [t for t in all_ts if int(t) <= a.start_t]
    if not ts:
        ts = all_ts[-1:]
    print('[diff] 去噪步数 %d（从 t=%d 起，原始 %d 步）'
          % (len(ts), int(ts[0]), len(all_ts)), flush=True)
    dev = ibe.device
    t0 = time.time()
    n_done = 0

    # ---- 条件准备 ----
    COND = None
    if a.prompt:
        if int(ck.get('cond_dim', 0)) <= 0:
            raise SystemExit('[diff] ckpt 是无条件模型（cond_dim=0），不能配 --prompt')
        from scripts.lib.prompt_cond import encode as _encode
        plist = a.prompt
        e = _encode(plist)
        if len(plist) == 1:
            COND = e[:1].expand(a.n, -1).contiguous()
            print('[diff] 提示词（%d 张共用）: %s' % (a.n, plist[0][:80]), flush=True)
        else:
            if len(plist) != a.n:
                raise SystemExit('[diff] 给了 %d 条提示词但 --n=%d，必须相等' % (len(plist), a.n))
            COND = e.contiguous()
            print('[diff] 提示词 %d 条，逐图对应' % len(plist), flush=True)
    elif a.cond_pt:
        if int(ck.get('cond_dim', 0)) <= 0:
            raise SystemExit('[diff] ckpt 是无条件模型（cond_dim=0），不能配 --cond-pt')
        ALLC = torch.load(a.cond_pt, map_location='cpu', weights_only=False).float()
        if a.cond_index is not None:
            COND = ALLC[a.cond_index:a.cond_index + 1].expand(a.n, -1).contiguous()
        else:
            s0 = a.cond_start
            COND = ALLC[s0:s0 + a.n].contiguous()
        print('[diff] 条件 %s  用第 %s 行  cfg_scale=%.2f'
              % (os.path.basename(a.cond_pt),
                 a.cond_index if a.cond_index is not None else "%d..%d" % (a.cond_start, a.cond_start + a.n - 1),
                 a.cfg_scale), flush=True)
    use_cfg = COND is not None and a.cfg_scale != 1.0

    # ★ DML 有一个实测的坑：它的【已提交显存堆随每次迭代净增且不回收】。
    #   实测 LoopedDiT(n=16) 每迭代 +394 MB，到 ~7.6 GB 上限就 OOM
    #   （15 次迭代左右），而且只报一个没头没尾的 RuntimeError。
    #   这里接住它，给出可操作的提示，而不是抛裸栈。
    try:
        with torch.no_grad():
            x = ibe.to_dev(x)
            cd = ibe.to_dev(COND) if COND is not None else None
            zero = torch.zeros_like(cd) if use_cfg else None
            for t in ts:
                tt = torch.full((a.n,), int(t), device=dev, dtype=torch.long)
                pred = ibe.forward_dev(x, tt, cd)
                if use_cfg:
                    # classifier-free guidance：eps = eps_u + s*(eps_c - eps_u)
                    pu = ibe.forward_dev(x, tt, zero)
                    pred = pu + a.cfg_scale * (pred - pu)
                x = sch.step(pred, t, x).prev_sample
                n_done += 1
            x = x.cpu()
    except RuntimeError as e:
        if 'video memory' in str(e) or 'allocate' in str(e):
            raise SystemExit(
                "\n[diff] ✗ 核显显存耗尽，在第 %d/%d 步。\n"
                "       原因（实测）：DirectML 的已提交堆每次迭代净增且不回收，\n"
                "       上限约 7.6 GB。LoopedDiT 在 n=16 时约 +394 MB/迭代。\n"
                "       办法：\n"
                "         1) 减小 --n（实测 LoopedDiT: n=8 可跑 19 步，n=16 只能 ~15 步）\n"
                "         2) 减少 --ddim-steps 或调低 --start-t\n"
                "         3) 改用 --infer-device cpu\n"
                "       注意：不要用 torch_directml.disable_tiled_resources(True)，实测更糟。\n"
                % (n_done, len(ts)))
        raise
    dt = time.time() - t0
    extra = "" if COND is None else "  cfg=%.2f" % a.cfg_scale
    print('[diff] 采样 %d 张  %.1fs (%.2f s/张, %d 步)  设备=%s%s'
          % (a.n, dt, dt / a.n, a.ddim_steps, ibe.desc, extra), flush=True)
    ibe.close()   # 采样完成即释放核显副本，后面 VQ 解码是 CPU 的

    # 训练时把潜变量变换到单位尺度，采样必须做【完全一致的逆变换】才是 decoder 认的尺度。
    # 2026-09-23 起默认是逐通道标准化（旧 ckpt 只有单标量 scale，走兼容分支）。
    from scripts.lib.latent_norm import meta_from_ckpt, denormalize, describe
    nmeta = meta_from_ckpt(ck)
    print('[diff] 反归一化: %s' % describe(nmeta), flush=True)
    x = denormalize(x, nmeta)

    vk = torch.load(a.vqvae, map_location='cpu', weights_only=False)
    vq, _ = vqvae_from_ckpt(vk)
    vq.eval()
    with torch.no_grad():
        imgs = torch.cat([vq.dec(x[s:s + 8]).clamp(-1, 1) for s in range(0, a.n, 8)], 0)

    arr = ((imgs.permute(0, 2, 3, 1).numpy() + 1) / 2 * 255).round().astype(np.uint8)

    # ★ 原版这里写死 side = round(sqrt(n))，只在 n 是完全平方数时成立：
    #   n=8 时 side=3，代码要 9 张但只有 8 张 -> IndexError。改成通用网格，
    #   不足的格子填白，任意 n 都能出图。
    import math
    cols = max(1, int(math.ceil(a.n ** 0.5)))
    rows_n = max(1, int(math.ceil(a.n / cols)))
    Hh, Ww = arr.shape[1], arr.shape[2]
    canvas = np.full((rows_n * Hh, cols * Ww, 3), 255, np.uint8)
    for i in range(a.n):
        r, c = divmod(i, cols)
        canvas[r * Hh:(r + 1) * Hh, c * Ww:(c + 1) * Ww] = arr[i]
    Image.fromarray(canvas).save(a.out)
    print('[diff] -> %s  (%d 张, %dx%d 网格)' % (a.out, a.n, rows_n, cols), flush=True)

    # ---- 模式覆盖：生成 vs 真实验证集，各自最近码的直方图 ----
    try:
        W = vq.vq.codebook.weight.data.float()
        Wn = F.normalize(W, dim=1)
        V = W.shape[0]

        def hist(zs):
            zf = zs.permute(0, 2, 3, 1).reshape(-1, zs.shape[1]).float()
            idx = (F.normalize(zf, dim=1) @ Wn.T).argmax(1)
            return torch.bincount(idx, minlength=V).float() / max(1, idx.numel())

        L = torch.load(a.latents, map_location='cpu', weights_only=False)
        lo, hi = 27658, min(29113, L.shape[0])
        p_real = hist(L[lo:hi].float())
        p_gen = hist(x)
        tv = 0.5 * float((p_gen - p_real).abs().sum())
        kl = float((p_real.clamp_min(1e-9) *
                    (p_real.clamp_min(1e-9) / p_gen.clamp_min(1e-9)).log()).sum())
        ent = float(-(p_gen.clamp_min(1e-9) * p_gen.clamp_min(1e-9).log()).sum())
        ent_r = float(-(p_real.clamp_min(1e-9) * p_real.clamp_min(1e-9).log()).sum())
        print('[diff] 模式覆盖: 生成熵=%.3f  真实熵=%.3f  TV=%.4f  KL(real||gen)=%.4f'
              % (ent, ent_r, tv, kl))
        print('[diff]   生成用到 %d/%d 码, 真实用到 %d/%d 码'
              % (int((p_gen > 0).sum()), V, int((p_real > 0).sum()), V))
    except Exception as e:
        print('[diff] 模式覆盖计算失败: %s' % str(e)[:120])


if __name__ == '__main__':
    main()
