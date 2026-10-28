# -*- coding: utf-8 -*-
"""rft_feasibility.py — 拒绝采样微调(RFT)值不值得做？先问"best-of-N 能挑出多好的"。

逻辑：RFT 的前提是"模型能采到好样本，只是概率低"。若 best-of-192 仍然很糊，
说明模型根本产不出好样本 ⇒ RFT 无从谈起，得换路子（改损失/改架构）。

用已有数据，零成本：
  · GANDATA 里有 192 张 FREE 样本及其 token
  · 判据用【对比度 std 与真实图的接近程度】（平凡基线已有 78.7% 判别力）
  · 给出 N=1..192 的"best-of-N 对比度"曲线，以及 best-16 的看图
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
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
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def save_grid(img, path, cols=4):
    n = img.shape[0]
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    S = img.shape[-1]
    canvas = Image.new('RGB', (cols * S, rows * S), (20, 20, 20))
    for i in range(n):
        r, c = divmod(i, cols)
        t = ((img[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
        canvas.paste(Image.fromarray(t), (c * S, r * S))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--top', type=int, default=16)
    a = ap.parse_args()

    z = torch.load(a.data, map_location='cpu', weights_only=False)
    real, free = z['real'].float(), z['free'].float()
    step = z.get('step')
    print('[rft] step=%s  real=%d  free=%d' % (step, real.shape[0], free.shape[0]))

    std_real = real.std(dim=(1, 2, 3)).mean().item()
    std_free = free.std(dim=(1, 2, 3))
    print('[rft] 真实图对比度均值 %.4f   自由采样 %.4f (%.1f%%)'
          % (std_real, std_free.mean(), 100 * std_free.mean() / std_real))

    # ---- best-of-N 曲线：随机抽 N 个，取其中最接近真实对比度的 ----
    torch.manual_seed(0)
    print()
    print('[rft] === best-of-N 能挑到多好的对比度（越接近 100%% 越好）===')
    print('  %6s %12s %12s' % ('N', 'best std', 'vs real'))
    n_all = std_free.shape[0]
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 192):
        if N > n_all:
            break
        bests = []
        for _ in range(64):                      # 多次重抽，报均值
            idx = torch.randperm(n_all)[:N]
            s = std_free[idx]
            bests.append(s[(s - std_real).abs().argmin()].item())
        b = sum(bests) / len(bests)
        print('  %6d %12.4f %11.1f%%' % (N, b, 100 * b / std_real))

    # ---- 取全局 best-k 看图 ----
    order = (std_free - std_real).abs().argsort()
    print()
    print('[rft] 最接近真实对比度的 %d 张的 std: %s' % (a.top, [round(std_free[i].item(), 3) for i in order[:6]]))
    print('[rft] 最差的 %d 张:            %s' % (a.top, [round(std_free[i].item(), 3) for i in order[-6:]]))
    save_grid(free[order[:a.top]], a.out_prefix + '_BEST.png')
    save_grid(free[order[-a.top:]], a.out_prefix + '_WORST.png')
    save_grid(real[:a.top], a.out_prefix + '_REAL.png')
    print('[rft] -> %s_{BEST,WORST,REAL}.png' % a.out_prefix)

    # ---- 关键判读：即使最好的，高频占比是否仍低于真实？----
    def hf(x):
        lp = F.interpolate(F.avg_pool2d(x, 4), size=x.shape[-2:], mode='nearest')
        return (((x - lp) ** 2).mean(dim=(1, 2, 3)) / (x.var(dim=(1, 2, 3)) + 1e-8))
    hf_real, hf_free = hf(real), hf(free)
    hf_best = hf(free[order[:a.top]])
    print()
    print('[rft] 高频占比：真实 %.4f | 自由采样均值 %.4f | **best-%d 均值 %.4f**'
          % (hf_real.mean(), hf_free.mean(), a.top, hf_best.mean()))
    print('[rft]   ⇒ 若 best-%d 的高频仍远低于真实，说明"挑"解决不了，得改损失/架构'
          % a.top)

    # ================= 决定性一问：模型【有没有可能】产出高频正常的样本？ =================
    # 按"高频占比"而不是"对比度"来挑，然后看 best-of-N 曲线。
    # 若 best-of-192 的高频仍远低于真实 ⇒ 模型根本产不出正确的高频结构
    #   ⇒ 任何"拒绝采样/挑最好"的方案都无效，必须改训练目标或架构。
    target = hf_real.mean().item()
    print()
    print('[rft] === 决定性：按【高频占比】挑，best-of-N 能到多接近真实（真实=%.4f）==='
          % target)
    print('  %6s %12s %12s' % ('N', 'best HF', 'vs real'))
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 192):
        if N > hf_free.shape[0]:
            break
        bests = []
        for _ in range(64):
            idx = torch.randperm(hf_free.shape[0])[:N]
            s = hf_free[idx]
            # 挑【不超过真实】里最接近真实的（超过真实可能是伪影，不算"好"）
            cand = s[s <= target]
            bests.append((cand.max() if cand.numel() else s.min()).item())
        print('  %6d %12.4f %11.1f%%' % (N, sum(bests) / len(bests),
                                         100 * (sum(bests) / len(bests)) / target))
    print('  ⇒ 若 N 很大时仍停在 ~50%% ⇒ 模型【结构上产不出】正确的高频，')
    print('     拒绝采样无效，必须改损失/架构（如 scheduled sampling、对抗、生成式解码器）。')

    # 按高频挑出来的最好 16 张，也存图看
    o2 = hf_free.argsort(descending=True)
    print()
    print('[rft] 高频占比最高的 6 张: %s   (真实均值 %.4f)'
          % ([round(hf_free[i].item(), 4) for i in o2[:6]], target))
    save_grid(free[o2[:a.top]], a.out_prefix + '_HFBEST.png')
    print('[rft] -> %s_HFBEST.png' % a.out_prefix)


if __name__ == '__main__':
    main()
