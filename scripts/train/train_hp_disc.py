# -*- coding: utf-8 -*-
"""train_hp_disc.py — 训练并保存【高通判别器】，供云端 RFT 挑选使用。

用法（本地，约 1-2 分钟）：
  python train_hp_disc.py --data <GANDATA.pt> --out hp_disc.pt

为什么判别器喂高通残差而不是原图：
  全图 D 在【低通消融】下准确率一点不掉（96.3%）⇒ **它不看纹理，只看对比度/全局结构**。
  用它做挑选/对抗只会把对比度推上去，治不了细节。
  喂 `x - lowpass(x)` 后，判别力【反而更高】（98.5%），且强迫它只在纹理域判断。
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
from scripts.lib.train_vqvae_gan import PatchDiscriminator, hinge_d
from scripts.lib.train_diag_disc import t_highpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--ndf', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    z = torch.load(a.data, map_location='cpu', weights_only=False)
    real, free = z['real'].float(), z['free'].float()
    if a.size != real.shape[-1]:
        real = F.interpolate(real, size=a.size, mode='bilinear', align_corners=False)
        free = F.interpolate(free, size=a.size, mode='bilinear', align_corners=False)
    R, Fk = t_highpass(real), t_highpass(free)
    print('[hpd] real %s fake %s  (高通域)' % (tuple(R.shape), tuple(Fk.shape)))

    torch.manual_seed(a.seed)
    d = PatchDiscriminator(in_ch=3, ndf=a.ndf, n_layers=3)
    opt = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    n = R.shape[0]
    for it in range(a.steps):
        i = torch.randint(0, n, (16,))
        j = torch.randint(0, n, (16,))
        opt.zero_grad(set_to_none=True)
        loss = hinge_d(d(R[i]), d(Fk[j]))
        loss.backward()
        opt.step()
        if (it + 1) % 300 == 0:
            print('  step %d  D-loss %.4f' % (it + 1, loss.item()), flush=True)
    d.eval()

    # 自检：分离度与 TF 归属
    with torch.no_grad():
        sr, sf = d(R).mean(dim=(1, 2, 3)), d(Fk).mean(dim=(1, 2, 3))
        st = d(t_highpass(F.interpolate(z['tf'].float(), size=a.size,
                                        mode='bilinear', align_corners=False))).mean(dim=(1, 2, 3))
    acc = ((sr > 0).sum().item() + (sf < 0).sum().item()) / (sr.numel() + sf.numel())
    print('[hpd] 真图 %.3f | TF %.3f | FREE %.3f  ⇒ 准确率 %.1f%%'
          % (sr.mean(), st.mean(), sf.mean(), acc * 100))

    torch.save(dict(state=d.state_dict(), ndf=a.ndf, size=a.size, in_ch=3,
                    mean_real=sr.mean().item(), mean_free=sf.mean().item(),
                    acc=acc, note='high-pass PatchGAN, trained on real vs free-running V32 decodes'),
               a.out)
    print('[hpd] -> %s (%.2f MB)' % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
