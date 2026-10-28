# -*- coding: utf-8 -*-
"""diag_diff_mse_by_t.py — ε 预测误差【按时间步分箱】，定位"生成不出图"的真因。

推理链：
  * val 总 MSE=0.0775 看着很好（预测 0 的基线是 1.0，等于解释了 92% 的方差）；
  * t0=800 的"加噪再还原"MSE 只有 0.1477（说明中低 t 的去噪链是对的）；
  * 但从【纯噪声】起步生成出来是彩虹噪点。
⇒ 唯一能同时解释这三件事的假设是：**模型在高 t 段（靠近 1000）的 ε 预测很差**。
  "加噪再还原"在 t0=800 成功，可能只是因为 t<=800 段好；
  而生成必须先走完顶部那段。

本脚本按 t 分箱量 MSE，并对比 EMA 权重与原始权重。
判据：若高 t 箱的 MSE 接近 1.0（= 只会预测 0），则顶部段没学会 ⇒ 生成必然失败。
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
import argparse, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.train_latent_diffusion import build_unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=r'.autopilot\diff_cmse\ckpt_5000.pt')
    ap.add_argument('--latents', default=r'.autopilot\latents_cmse.pt')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from diffusers import DDPMScheduler
    sch = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                        clip_sample=False)
    ac = sch.alphas_cumprod

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    S, Cin = int(ck.get('sample_size', 16)), int(ck.get('in_ch', 64))
    scale = float(ck.get('scale', 1.0))
    L = torch.load(a.latents, map_location='cpu', weights_only=False).float() / scale
    x0 = L[27658:27658 + a.n]
    print('[diag] ckpt step=%s  scale=%.4f  val 样本 %d' % (ck.get('step'), scale, len(x0)))

    bins = [(0, 100), (100, 200), (200, 400), (400, 600), (600, 800), (800, 900), (900, 1000)]
    for tag, sd in (('EMA', ck.get('ema')), ('raw', ck['model'])):
        if not sd:
            continue
        model = build_unet([int(x) for x in str(ck.get('ch', '128,256,256')).split(',')],
                           sample_size=S, in_ch=Cin)
        model.load_state_dict(sd, strict=False)
        model.eval()
        print()
        print('=== 权重: %s ===' % tag)
        print('%-12s %10s %10s %10s' % ('t 区间', 'MSE(eps)', 'baseline1.0', 'eps std'))
        print('-' * 46)
        for lo, hi in bins:
            g = torch.Generator().manual_seed(lo)
            tot, cnt, es = 0.0, 0, []
            with torch.no_grad():
                for s in range(0, len(x0), 32):
                    xb = x0[s:s + 32]
                    eps = torch.randn(xb.shape, generator=g)
                    t = torch.randint(lo, hi, (xb.shape[0],), generator=g)
                    xt = ac[t].sqrt().view(-1, 1, 1, 1) * xb + (1 - ac[t]).sqrt().view(-1, 1, 1, 1) * eps
                    pred = model(xt, t).sample
                    tot += float(F.mse_loss(pred, eps)) * xb.shape[0]
                    cnt += xb.shape[0]
                    es.append(float(eps.std()))
            print('%-12s %10.4f %10.1f %10.3f' % ('%d-%d' % (lo, hi), tot / cnt,
                                                  (tot / cnt) / 1.0, sum(es) / len(es)))
    print()
    print('判据：MSE 越小越好；=1.0 表示"只会预测 0"。')
    print('      若 900-1000 箱的 MSE 接近 1.0 ⇒ 顶部没学会，从纯噪声生成必然失败。')


if __name__ == '__main__':
    main()
