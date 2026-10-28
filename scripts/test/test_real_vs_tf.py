# -*- coding: utf-8 -*-
"""test_real_vs_tf.py — 更正一个错误结论：TF 到底能不能与真图区分？

我之前的错误：
  拿一个在 "真图 vs FREE" 上训练的判别器去测 TF，得到 52.9%，就断言
  "TF 与真图感知上无法区分"。
  但那个测法回答的是"**TF 落在真图那一侧吗**"，不是"TF 和真图像不像"。
  FREE 离真图极远（抽象色块 vs 清晰照片）⇒ 判别边界很松 ⇒ 糊掉的 TF 当然还在真图一侧。
  **看图就知道 TF 明显比 TRUE 糊。**

正确测法：**直接训练 "真图 vs TF" 的判别器**，看它能不能分开。
  能分开（准确率 >> 50%）⇒ TF 与真图是可区分的，我之前的结论作废。
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
from scripts.lib.train_vqvae_gan import PatchDiscriminator, hinge_d
from scripts.lib.train_diag_disc import t_highpass, hf_energy


def train_d(real, fake, steps=800, ndf=32, seed=0, bs=16, lr=2e-4):
    torch.manual_seed(seed)
    d = PatchDiscriminator(in_ch=3, ndf=ndf, n_layers=3)
    opt = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    for _ in range(steps):
        i = torch.randint(0, real.shape[0], (bs,))
        j = torch.randint(0, fake.shape[0], (bs,))
        opt.zero_grad(set_to_none=True)
        hinge_d(d(real[i]), d(fake[j])).backward()
        opt.step()
    d.eval()
    return d


@torch.no_grad()
def acc(d, a, b):
    sa, sb = d(a).mean(dim=(1, 2, 3)), d(b).mean(dim=(1, 2, 3))
    return ((sa > 0).sum().item() + (sb < 0).sum().item()) / (sa.numel() + sb.numel())


def run(real, other, name, size=128, steps=800, split=0.65):
    R, O = real, other
    if R.shape[-1] != size:
        R = F.interpolate(R, size=size, mode='bilinear', align_corners=False)
        O = F.interpolate(O, size=size, mode='bilinear', align_corners=False)
    n = R.shape[0]; nt = int(n * split)
    for dom, fn in (('全图', lambda x: x), ('高通', t_highpass)):
        d = train_d(fn(R[:nt]), fn(O[:nt]), steps=steps)
        a = acc(d, fn(R[nt:]), fn(O[nt:]))
        print('  %-4s  real vs %-5s  准确率 %5.1f%%' % (dom, name, a * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--steps', type=int, default=800)
    a = ap.parse_args()
    z = torch.load(a.data, map_location='cpu', weights_only=False)
    real, tf, free = z['real'].float(), z['tf'].float(), z['free'].float()
    print('[fix] n=%d  真实图高频占比 %.4f | TF %.4f | FREE %.4f'
          % (real.shape[0], hf_energy(F.interpolate(real, size=128, mode='bilinear')).mean(),
             hf_energy(F.interpolate(tf, size=128, mode='bilinear')).mean(),
             hf_energy(F.interpolate(free, size=128, mode='bilinear')).mean()))
    print()
    print('[fix] === 各自【直接】训练 real-vs-X 判别器（正确测法）===')
    run(real, free, 'FREE', steps=a.steps)
    run(real, tf, 'TF', steps=a.steps)
    print()
    print('[fix] 判读：TF 那一行若明显 >> 50%% ⇒ TF 与真图【可区分】')
    print('      ⇒ 我之前"TF 感知上无法区分"的结论作废（测法本身错了）。')


if __name__ == '__main__':
    main()
