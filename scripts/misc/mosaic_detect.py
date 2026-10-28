# -*- coding: utf-8 -*-
"""mosaic_detect.py - 马赛克 / 像素化（censorship mosaic）自动检测。

为什么需要它
------------
马赛克是**不可逆**的：像素被平均后信息已经丢失，没有任何算法能还原。
如果训练集里混入马赛克图，扩散/AR 模型会把它当成"正常纹理"学进去 ——
这是唯一一类"宁可错杀"的脏数据。

判据
----
马赛克 = 规则网格上的像素块平均。它留下两个指纹：
  1. **块边界跳变**：在 x % b == 0 的位置，相邻像素差显著大于块内部
  2. **块内平坦**：块内方差异常低
本工具对一系列候选块尺寸 b 计算「边界梯度 / 内部梯度」比值，取最大者作为
块状度分数，并在图像的局部 patch 上取最大值（马赛克通常只覆盖局部区域，
全局平均会把信号稀释掉）。

用法
----
  # 自检：合成马赛克，验证检测器确实能抓到
  python mosaic_detect.py --selftest

  # 扫描一个目录
  python mosaic_detect.py --data <dir> --out mosaic.jsonl --res 512 --threshold 1.30

  # 先只输出分数分布，用来标定阈值
  python mosaic_detect.py --data <dir> --out mosaic.jsonl --report-only
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def load_gray(path, res):
    im = Image.open(path)
    try:
        im.draft("L", (res, res))
    except Exception:
        pass
    im = im.convert("L")
    w, h = im.size
    if w != h:                      # 中心裁剪成方形，避免长宽比干扰块网格
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if im.size != (res, res):
        im = im.resize((res, res), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


@torch.no_grad()
def _blockiness_map(g, bmin=3, bmax=32):
    """g: (N,P,1,h,w) -> (N,P) 块状度分数 与 对应块尺寸。

    用 FFT 找梯度图的周期峰值：马赛克在 b 像素周期上有规律性跳变，
    会在频谱 k = W/b 处形成峰值。**与网格相位无关**（相位只影响复数相位，
    不影响幅度谱），这比固定掩码稳健得多。
    分数 = 峰值幅度 / 该 patch 频谱中位数（即"周期性显著度"）。
    """
    N, P, _, h, w = g.shape
    dx = (g[..., 1:] - g[..., :-1]).abs()          # 水平梯度
    dy = (g[..., 1:, :] - g[..., :-1, :]).abs()    # 垂直梯度

    def band_score(d, axis_len):
        F = torch.fft.rfft(d, dim=-1).abs()        # (N,P,1,h,k)
        s = F.mean(dim=(2, 3))                     # (N,P,k) 对行取平均
        k = s.shape[-1]
        # 对应块尺寸 b 的频率索引: kfreq = axis_len / b
        lo = max(1, int(axis_len / bmax))
        hi = min(k - 1, int(axis_len / bmin) + 1)
        if hi <= lo:
            return None, None
        band = s[:, :, lo:hi + 1]
        peak, pi = band.max(dim=-1)                # (N,P)
        med = s.median(dim=-1).values              # (N,P)
        return peak / (med + 1e-7), pi + lo

    sx, kx = band_score(dx, w - 1)
    out_s, out_k = sx, kx
    if sx is not None:
        # 频率索引 -> 块尺寸
        out_k = torch.where(kx > 0, ((w - 1) / kx.clamp(min=1)).round(), torch.zeros_like(kx))
    sy, ky = band_score(dy, h - 1)
    if sy is not None:
        ky_b = ((h - 1) / ky.clamp(min=1)).round()
        take = sy > out_s
        out_s = torch.where(take, sy, out_s)
        out_k = torch.where(take, ky_b, out_k)
    return out_s, out_k


@torch.no_grad()
def mosaic_score(x, grid=4):
    """x: (N,3,H,W) float32 in [0,1]。返回 (score[N], block[N], flat[N])。

    把图切成 grid x grid 个 patch 分别算块状度再取最大 —— 马赛克通常只覆盖
    局部区域，全局平均会把信号稀释掉。
    """
    g = (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).unsqueeze(1)
    N, _, H, W = g.shape
    ph, pw = H // grid, W // grid
    patches = []
    for i in range(grid):
        for j in range(grid):
            patches.append(g[:, :, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw])
    gp = torch.stack(patches, dim=1)               # (N,P,1,ph,pw)
    s, b = _blockiness_map(gp)                     # (N,P)
    score = s.max(dim=1).values
    blk = b.gather(1, s.argmax(dim=1, keepdim=True)).squeeze(1)
    # 局部方差（16x16 窗口的最小值），马赛克区域会被抹平
    v = F.avg_pool2d(g, 16, stride=8)
    v2 = F.avg_pool2d(g * g, 16, stride=8)
    var = (v2 - v * v).clamp(min=0)
    flat = var.amin(dim=(1, 2, 3))
    return score, blk, flat


def pixelate(x, b, box=None):
    """把 x (N,3,H,W) 指定区域打成 b x b 马赛克（用于自检）。"""
    y = x.clone()
    N, C, H, W = y.shape
    x0, y0, x1, y1 = box or (W // 4, H // 4, 3 * W // 4, 3 * H // 4)
    reg = y[:, :, y0:y1, x0:x1]
    h, w = reg.shape[-2:]
    hb, wb = h // b, w // b
    reg = reg[:, :, :hb * b, :wb * b]
    reg = reg.reshape(N, C, hb, b, wb, b).mean(dim=(3, 5), keepdim=True)
    reg = reg.expand(N, C, hb, b, wb, b).reshape(N, C, hb * b, wb * b)
    y[:, :, y0:y0 + hb * b, x0:x0 + wb * b] = reg
    return y


def selftest(res=512, n=6):
    """合成马赛克验证检测器：干净图 vs 打了马赛克的图，分数必须拉开。"""
    torch.manual_seed(0)
    # 合成"类照片"图：低频结构 + 轻微噪声（真实照片的平滑区域就是这个统计特性）
    base = torch.rand(n, 3, res // 16, res // 16)
    base = F.interpolate(base, size=(res, res), mode='bicubic', align_corners=False)
    base = (base + 0.02 * torch.randn(n, 3, res, res)).clamp(0, 1)
    print(f'{"块尺寸":>8} {"干净分数":>10} {"马赛克分数":>12} {"倍数":>8} {"判定":>6} {"检出块":>8}')
    ok = 0
    total = 0
    for b in (4, 6, 8, 10, 12, 16, 20, 24):
        clean_s, _, _ = mosaic_score(base)
        mos = pixelate(base, b)
        mos_s, mos_blk, _ = mosaic_score(mos)
        c = float(clean_s.mean()); m = float(mos_s.mean())
        ratio = m / max(c, 1e-6)
        hit = ratio > 1.5
        if hit: ok += 1
        total += 1
        print(f'{b:>8} {c:>10.3f} {m:>12.3f} {ratio:>7.2f}x {"OK" if hit else "MISS":>6} {float(mos_blk.mean()):>8.0f}')
    print(f'\n自检结果: {ok}/{total} 个块尺寸被检出（要求马赛克分数 >= 干净图的 1.5 倍）')
    return ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--out', default='')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--threshold', type=float, default=1.30,
                    help='块状度阈值，超过判定为疑似马赛克')
    ap.add_argument('--report-only', action='store_true', help='只输出分数分布')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.selftest:
        sys.exit(0 if selftest(a.res) else 1)
    if not a.data:
        ap.error('需要 --data 或 --selftest')

    paths = list_images(a.data)
    if a.limit:
        paths = paths[:a.limit]
    rows = []
    t0 = time.perf_counter()
    for s in range(0, len(paths), a.batch):
        chunk = paths[s:s + a.batch]
        arrs = []
        for p in chunk:
            try:
                arrs.append(load_gray(p, a.res))
            except Exception:
                arrs.append(np.zeros((a.res, a.res), np.float32))
        x = torch.from_numpy(np.stack(arrs)).unsqueeze(1).repeat(1, 3, 1, 1)
        sc, blk, flat = mosaic_score(x)
        for i, p in enumerate(chunk):
            rows.append(dict(src=p, score=round(float(sc[i]), 4), block=int(blk[i]),
                             flatness=round(float(flat[i]), 6),
                             mosaic=bool(float(sc[i]) >= a.threshold)))
        if (s // a.batch) % 10 == 0:
            el = time.perf_counter() - t0
            print(f'[mosaic] {min(s+a.batch,len(paths))}/{len(paths)} {el:.1f}s', flush=True)

    dt = time.perf_counter() - t0
    sc = sorted(r['score'] for r in rows)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n = len(sc)
    if n:
        q = lambda p: sc[min(n - 1, int(p * n))]
        print(f'[mosaic] n={n} 耗时 {dt:.1f}s ({dt/max(1,n)*1000:.1f} ms/张)')
        print(f'[mosaic] 分数分布: min={sc[0]:.3f} p25={q(.25):.3f} 中位={q(.5):.3f} '
              f'p75={q(.75):.3f} p90={q(.90):.3f} p99={q(.99):.3f} max={sc[-1]:.3f}')
        print(f'[mosaic] 阈值 {a.threshold} 判定疑似马赛克: '
              f'{sum(1 for r in rows if r["mosaic"])} / {n}'
              f'  ({sum(1 for r in rows if r["mosaic"])/n*100:.1f}%)')
        # 判为马赛克时的块尺寸分布，能反推马赛克强度
        import collections
        bb = collections.Counter(r['block'] for r in rows if r['mosaic'])
        if bb:
            print(f'[mosaic] 检出块尺寸分布: {dict(sorted(bb.items()))}')
    if a.out:
        print(f'[mosaic] -> {a.out}')


if __name__ == '__main__':
    main()
