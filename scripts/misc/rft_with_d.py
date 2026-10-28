# -*- coding: utf-8 -*-
"""rft_with_d.py — 用【高通判别器】当挑选判据，看能否在小 N 下找到真正好的样本。

为什么这个测试关键：
  · 用原始统计量挑：高频占比高的全是【噪声】（真实 0.048，最高的样本 0.219 = 4.5×）
    ⇒ 朴素统计量会把噪声当"好"
  · 但高通判别器不是在看"高频有多少"，而是在看"**纹理像不像真的**"
    ⇒ 它理应对噪声免疫（噪声不像真实纹理）
  · 若 D 能在 N=16 就挑出纹理正常的样本 ⇒ RFT 从"需要 N=128、131k 次采样、124 小时"
    变成经济可行；若 D 挑出来的仍是噪声，则这条路也否。

用法：在 train split 上训高通 D，对 test split 的 FREE 样本打分，取 top-16 看。
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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.train_vqvae_gan import PatchDiscriminator, hinge_d
from scripts.lib.train_diag_disc import t_highpass, hf_energy


def train_d(real, fake, steps, ndf=32, seed=0, bs=16, lr=2e-4):
    torch.manual_seed(seed)
    d = PatchDiscriminator(in_ch=real.shape[1], ndf=ndf, n_layers=3)
    opt = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    for _ in range(steps):
        i = torch.randint(0, real.shape[0], (min(bs, real.shape[0]),))
        j = torch.randint(0, fake.shape[0], (min(bs, fake.shape[0]),))
        opt.zero_grad(set_to_none=True)
        hinge_d(d(real[i]), d(fake[j])).backward()
        opt.step()
    d.eval()
    return d


def save_grid(img, path, cols=4):
    n = img.shape[0]; cols = min(cols, n); rows = (n + cols - 1) // cols; S = img.shape[-1]
    canvas = Image.new('RGB', (cols * S, rows * S), (20, 20, 20))
    for i in range(n):
        r, c = divmod(i, cols)
        t = ((img[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
        canvas.paste(Image.fromarray(t), (c * S, r * S))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--train-frac', type=float, default=0.65)
    ap.add_argument('--top', type=int, default=16)
    ap.add_argument('--out-prefix', required=True)
    a = ap.parse_args()

    z = torch.load(a.data, map_location='cpu', weights_only=False)
    real, free, tf = z['real'].float(), z['free'].float(), z['tf'].float()
    if a.size != real.shape[-1]:
        real = F.interpolate(real, size=a.size, mode='bilinear', align_corners=False)
        free = F.interpolate(free, size=a.size, mode='bilinear', align_corners=False)
        tf = F.interpolate(tf, size=a.size, mode='bilinear', align_corners=False)
    n = real.shape[0]; nt = int(n * a.train_frac)
    print('[dsel] step=%s n=%d train=%d test=%d' % (z.get('step'), n, nt, n - nt))

    # 高通域训练 D（real=真图残差, fake=自由采样残差）
    R, Fk = t_highpass(real), t_highpass(free)
    d = train_d(R[:nt], Fk[:nt], a.steps, seed=0)

    with torch.no_grad():
        s_free = d(Fk[nt:]).mean(dim=(1, 2, 3))       # 越高 = 越被当作真
        s_real = d(R[nt:]).mean(dim=(1, 2, 3))
        s_tf = d(t_highpass(tf)[nt:]).mean(dim=(1, 2, 3))
    print('[dsel] D 分数（高通域）：真图 %.3f | TF %.3f | FREE %.3f'
          % (s_real.mean(), s_tf.mean(), s_free.mean()))

    # 挑 D 分最高的 FREE
    order = s_free.argsort(descending=True)
    free_te = free[nt:]
    hf_free_te = hf_energy(free_te)
    hf_real = hf_energy(real).mean()
    pick = order[:a.top]
    print()
    print('[dsel] 真实图高频占比 %.4f' % hf_real)
    print('[dsel] test-FREE 高频占比：均值 %.4f' % hf_free_te.mean())
    print('[dsel] **D 挑出的 top-%d 的高频占比：%.4f**  (%.0f%% of real)'
          % (a.top, hf_free_te[pick].mean(), 100 * hf_free_te[pick].mean() / hf_real))
    print('[dsel] 对比：按【原始高频占比】挑的 top-%d：%.4f  (已知会挑到噪声)'
          % (a.top, hf_free_te[hf_free_te.argsort(descending=True)[:a.top]].mean()))
    print()
    print('[dsel] D 挑出的 top-6 的高频占比: %s'
          % [round(hf_free_te[i].item(), 4) for i in pick[:6]])
    save_grid(free_te[pick], a.out_prefix + '_DSEL.png')
    save_grid(real[:a.top], a.out_prefix + '_REAL.png')
    print('[dsel] -> %s_{DSEL,REAL}.png' % a.out_prefix)


if __name__ == '__main__':
    main()
