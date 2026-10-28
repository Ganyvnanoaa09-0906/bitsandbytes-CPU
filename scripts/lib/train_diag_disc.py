# -*- coding: utf-8 -*-
"""train_diag_disc.py — 诊断判别器：我们的输出到底有多"假"，以及 D 靠什么在判。

设计要点（这是【诊断】不是训练生成器）：
  1. 用 real vs FREE 训练一个 PatchGAN，然后在 **real vs TF** 上测试。
     ⇒ 若 D 分辨不出 TF（准确率≈50%），就从**感知度量**上独立印证了 §10.28：
       "病在采样、不在每步预测"。这是频域结论的独立复核。
  2. 平凡基线：单靠"对比度（std）阈值"能到多少准确率？
     ⇒ 若已接近 D 的水平，说明判据就是"发灰"，不需要 GAN。
  3. 消融：对输入做对比度归一化 / 低通 / 灰度，**各自重训一个 D**，
     看"去掉这个线索后，差异还在不在"。
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
import argparse, os, sys, time
import torch
import torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vqvae_gan import PatchDiscriminator, hinge_d


# ---------------- 消融变换 ----------------
def t_none(x):
    return x


def t_contrast(x):
    """把每张图的均值/方差变成和真实图一致（去"发灰"这个线索）"""
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-5)
    return (x - m) / s * 0.5      # 统一到 std=0.5、均值=0


def t_lowpass(x):
    """4x 平均池化再上采样 ⇒ 抹掉高频，只留低频结构"""
    h = F.avg_pool2d(x, 4)
    return F.interpolate(h, size=x.shape[-2:], mode='nearest')


def t_gray(x):
    g = x.mean(dim=1, keepdim=True)
    return g.repeat(1, x.shape[1], 1, 1)


def t_highpass(x):
    """只留高通残差 x - lowpass(x) ⇒ 强迫判别器【只能看纹理】。
    这是针对"低通消融后 D 准确率一点不掉"（⇒ D 不看纹理）的开药方：
    若在这个变换下 D 仍能显著判别，说明【高频域里确实有可分的信号】，
    高通判别器这条路成立；若掉到 ≈50%，说明这条也走不通。"""
    lp = F.interpolate(F.avg_pool2d(x, 4), size=x.shape[-2:], mode='nearest')
    r = x - lp
    # 残差本身量级很小，按整体 std 归一化，避免 D 因输入过小而学不动
    return r / r.std().clamp(min=1e-6)


def hf_energy(x):
    """高频能量占比：x - 低通(x) 的方差 / x 的方差（与 spectrum_gap 同源思路）"""
    lp = F.interpolate(F.avg_pool2d(x, 4), size=x.shape[-2:], mode='nearest')
    return ((x - lp) ** 2).mean(dim=(1, 2, 3)) / (x.var(dim=(1, 2, 3)) + 1e-8)


def train_d(real, fake, steps=600, ndf=32, seed=0, bs=16, lr=2e-4):
    torch.manual_seed(seed)
    d = PatchDiscriminator(in_ch=real.shape[1], ndf=ndf, n_layers=3)
    opt = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    n = real.shape[0]
    for it in range(steps):
        i = torch.randint(0, n, (min(bs, n),))
        j = torch.randint(0, fake.shape[0], (min(bs, fake.shape[0]),))
        opt.zero_grad(set_to_none=True)
        loss = hinge_d(d(real[i]), d(fake[j]))
        loss.backward()
        opt.step()
    d.eval()
    return d


@torch.no_grad()
def acc(d, a, b):
    """把 a 判为真、b 判为假；用 patch 平均分，报二分类准确率"""
    sa = d(a).mean(dim=(1, 2, 3))
    sb = d(b).mean(dim=(1, 2, 3))
    correct = (sa > 0).sum().item() + (sb < 0).sum().item()
    return correct / (sa.numel() + sb.numel())


@torch.no_grad()
def acc_std_threshold(real, fake):
    """平凡基线：只用"每张图的 std"当特征，阈值取两类中点"""
    sr = real.std(dim=(1, 2, 3))
    sf = fake.std(dim=(1, 2, 3))
    thr = (sr.mean() + sf.mean()) / 2
    return ((sr > thr).sum().item() + (sf < thr).sum().item()) / (sr.numel() + sf.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-frac', type=float, default=0.65)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--size', type=int, default=128,
                    help='判别器看到的边长（降采样以控制 CPU 时间；"发灰"是低频性质，128 足够）')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    z = torch.load(a.data, map_location='cpu', weights_only=False)
    real, tf, free = z['real'].float(), z['tf'].float(), z['free'].float()
    if a.size and a.size != real.shape[-1]:
        real = F.interpolate(real, size=a.size, mode='bilinear', align_corners=False)
        tf = F.interpolate(tf, size=a.size, mode='bilinear', align_corners=False)
        free = F.interpolate(free, size=a.size, mode='bilinear', align_corners=False)
    n = real.shape[0]
    nt = int(n * a.train_frac)
    R_tr, R_te = real[:nt], real[nt:]
    F_tr, F_te = free[:nt], free[nt:]
    T_te = tf[nt:]
    print('[diag] step=%s  n=%d (train %d / test %d)  img %s'
          % (z.get('step'), n, nt, n - nt, tuple(real.shape[1:])), flush=True)

    # ---------- 1) 基础统计（不用 D 也能看出来的差异）----------
    print()
    print('[diag] === 基础统计（均值/对比度/高频占比）===')
    print('  %-6s %10s %10s %10s' % ('class', 'mean', 'std', 'HF-ratio'))
    for nm, t in (('real', real), ('TF', tf), ('FREE', free)):
        print('  %-6s %10.4f %10.4f %10.4f'
              % (nm, t.mean(), t.std(dim=(1, 2, 3)).mean(), hf_energy(t).mean()))

    # ---------- 2) 平凡基线 ----------
    print()
    print('[diag] === 平凡基线：只用对比度(std)阈值 ===')
    print('  real vs FREE: %.1f%%' % (100 * acc_std_threshold(R_te, F_te)))
    print('  real vs TF  : %.1f%%' % (100 * acc_std_threshold(R_te, T_te)))

    # ---------- 3) 主判别器 ----------
    print()
    print('[diag] === PatchGAN 判别器（600 步，real vs FREE 训练）===')
    t0 = time.perf_counter()
    d = train_d(R_tr, F_tr, steps=a.steps, seed=a.seed)
    print('  训练耗时 %.0fs' % (time.perf_counter() - t0))
    base_free = acc(d, R_te, F_te)
    base_tf = acc(d, R_te, T_te)
    print('  ★ real vs FREE 准确率: %.1f%%   (越高=我们的输出越容易被识破)'
          % (100 * base_free))
    print('  ★ real vs TF   准确率: %.1f%%   (≈50%% ⇒ TF 与真图【感知上难以区分】)'
          % (100 * base_tf))

    # ---------- 4) 消融：去掉某个线索后，差异还在不在 ----------
    print()
    print('[diag] === 消融（各变换下【重训】一个 D；看差异是否仍在）===')
    print('  %-14s %14s %14s' % ('transform', 'acc(real/FREE)', 'acc(real/TF)'))
    for nm, fn in (('none', t_none), ('contrast-norm', t_contrast),
                   ('lowpass x4', t_lowpass), ('grayscale', t_gray),
                   ('gray+lowpass', lambda x: t_lowpass(t_gray(x))),
                   ('HIGHPASS only', t_highpass)):
        R2, F2, T2 = fn(real), fn(free), fn(tf)
        d2 = train_d(R2[:nt], F2[:nt], steps=a.steps, seed=a.seed + 1)
        print('  %-14s %13.1f%% %13.1f%%'
              % (nm, 100 * acc(d2, R2[nt:], F2[nt:]), 100 * acc(d2, R2[nt:], T2[nt:])), flush=True)

    print()
    print('[diag] 判读：')
    print('  · 若某消融后 acc 掉到 ≈50%% ⇒ **该线索就是全部差异来源**。')
    print('  · 若 real/TF 在所有消融下都 ≈50%% ⇒ 教师强制产物与真图无差别，')
    print('    病因确认为【采样漂移】，不是每步预测。')


if __name__ == '__main__':
    main()
