# -*- coding: utf-8 -*-
"""diag_diff_sampler.py — 分离"采样器错"与"模型没学好"。

现象：`diff_cmse` 在 step 5000 的 val MSE 已经降到 0.077（说明 eps 预测器学得不错），
但 DDIM 采样出来是【RGB 彩虹噪点】。这两件事不可能同时成立 ⇒ 采样链路有问题。

做法（决定性且便宜）：
  取【真实的验证集潜变量】x0，只在轻度噪声 t0 上加噪，再从 t0 反向去噪回来。
  - 若 t0=200 时能近似还原 x0 ⇒ 采样器是对的，彩虹噪点说明是别的问题（如 scale）；
  - 若连轻度加噪都还原不回来 ⇒ 采样器/日程/尺度 有硬 bug。
同时打印每一步潜变量的 std，看它是否发散。
"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, os, sys
import torch
import torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.train_latent_diffusion import build_unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=r'.autopilot\diff_cmse\ckpt_5000.pt')
    ap.add_argument('--latents', default=r'.autopilot\latents_cmse.pt')
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from diffusers import DDIMScheduler, DDPMScheduler
    train_sch = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                              clip_sample=False)
    samp = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                         clip_sample=False, set_alpha_to_one=False)
    ac = train_sch.alphas_cumprod

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    S, Cin = int(ck.get('sample_size', 16)), int(ck.get('in_ch', 64))
    model = build_unet([int(x) for x in str(ck.get('ch', '128,256,256')).split(',')],
                       sample_size=S, in_ch=Cin)
    model.load_state_dict(ck.get('ema') or ck['model'], strict=False)
    model.eval()
    scale = float(ck.get('scale', 1.0))
    print('[diag] ckpt step=%s  scale=%.4f' % (ck.get('step'), scale))

    L = torch.load(a.latents, map_location='cpu', weights_only=False).float() / scale
    x0 = L[27658:27662].clone()
    print('[diag] 真实潜变量(归一化后): mean=%.4f std=%.4f' % (float(x0.mean()), float(x0.std())))

    g = torch.Generator().manual_seed(0)
    print()
    print('=== A) 复现：不同 t0 的还原能力（当前采样器配置）===')
    for t0 in (200, 500, 800, 999):
        eps = torch.randn(x0.shape, generator=g)
        xt = ac[t0].sqrt() * x0 + (1 - ac[t0]).sqrt() * eps
        samp.set_timesteps(50)
        ts = [t for t in samp.timesteps if t <= t0]
        x = xt.clone()
        stds = []
        with torch.no_grad():
            for t in ts:
                pred = model(x, t).sample
                x = samp.step(pred, t, x).prev_sample
                stds.append(float(x.std()))
        mse = float(F.mse_loss(x, x0))
        print('[diag] t0=%-4d DDIM(%d步) 还原 MSE=%.4f  std: 起%.2f 末%.2f (max %.2f)'
              % (t0, len(ts), mse, float(xt.std()), stds[-1] if stds else float('nan'),
                 max(stds) if stds else float('nan')))

    # ---- B) 纯噪声起步：试不同的 clip 配置，看哪种不发散 ----
    print()
    print('=== B) t0=999 纯噪声起步，试不同 clip 配置 ===')
    print('%-34s %10s %8s %8s' % ('config', 'MSE', 'std末', 'used'))
    print('-' * 64)
    configs = [
        dict(clip_sample=False, set_alpha_to_one=False, tag='clip=False (当前)'),
        dict(clip_sample=True, clip_sample_range=4.0, set_alpha_to_one=False,
             tag='clip=True range=4.0'),
        dict(clip_sample=True, clip_sample_range=6.0, set_alpha_to_one=False,
             tag='clip=True range=6.0'),
        dict(clip_sample=True, clip_sample_range=1.0, set_alpha_to_one=False,
             tag='clip=True range=1.0'),
    ]
    for cf in configs:
        tag = cf.pop('tag')
        s2 = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', **cf)
        s2.set_timesteps(50)
        eps = torch.randn(x0.shape, generator=g)
        x = eps.clone()
        with torch.no_grad():
            for t in s2.timesteps:
                x = s2.step(model(x, t).sample, t, x).prev_sample
        # 生成不可能等于 x0，这里看的是"是否发散"和"码本使用是否合理"
        W = None
        print('%-34s %10s %8.2f %8s' % (tag, '-', float(x.std()), '-'))
        cf['tag'] = tag

    print()
    print('判据：std 末应保持在 ~1 附近。若 >3 ⇒ 采样发散（彩虹噪点就是这么来的）。')
    print('      MSE 这一列对纯噪声起步无意义（生成≠原图），只看 std 是否发散。')


if __name__ == '__main__':
    main()
