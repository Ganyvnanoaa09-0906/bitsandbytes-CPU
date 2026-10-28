# -*- coding: utf-8 -*-
"""mosaic_detect_v2.py - 马赛克/像素化检测 v2（块内方差判据 + 真实样本验证）。

v1 为什么失败
-------------
v1 用 FFT 找梯度周期峰值，两个毛病：
  1. 频段下界算成了 int(axis/32)=3，把 k=3 这种超低频也收进来，
     结果 5409/6419 张图的"最佳块尺寸"都报 42 —— 它在捡大尺度图像结构。
  2. 只在合成图上验证，没在真实照片上验证过。

v2 的判据
---------
马赛克的本质是**块内像素被平均**，所以真正的分水岭是：

              块边界跳变      块内标准差
  马赛克       很大           接近 0
  自然照片     中等           显著（纹理保留）
  JPEG 压缩    有 8x8 网格    仍然保留纹理   <-- v1 栽在这里

分数 = 边界跳变 / (块内标准差 + 0.005)，并做三件事：
  - **反解最优相位**：马赛克网格相位未知，从一维梯度剖面折叠求和找出最大
    响应所在相位，O(W) 覆盖全部相位（蛮力试相位是 O(b*W)）
  - 分 patch 取最大（马赛克通常只覆盖局部）
  - 可选排除 8/16/24（JPEG 网格尺寸）做对照

用法
----
  python mosaic_detect_v2.py --validate --data <真实图目录> --n 60 --res 512
  python mosaic_detect_v2.py --data <目录> --out scores.jsonl --res 512
"""
from __future__ import annotations
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
import argparse, json, os, random, time
import numpy as np
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
        im.draft('L', (res, res))
    except Exception:
        pass
    im = im.convert('L')
    w, h = im.size
    if w != h:
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if im.size != (res, res):
        im = im.resize((res, res), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def best_phase(prof, b):
    """从一维梯度剖面反解最优相位（0..b-1），O(W) 覆盖全部相位。"""
    n = len(prof)
    m = n // b
    if m < 2:
        return 0
    seg = prof[:m * b].reshape(m, b)
    return int(np.argmax(seg.sum(axis=0)))


def blockiness(g, b, phase_y=0, phase_x=0, border=1):
    """g: (H,W)。返回 (boundary_jump, interior_std)。"""
    H, W = g.shape
    y0 = phase_y % b
    x0 = phase_x % b
    ny = (H - y0) // b
    nx = (W - x0) // b
    if ny < 2 or nx < 2:
        return 0.0, 1.0
    t = g[y0:y0 + ny * b, x0:x0 + nx * b]
    tiles = np.ascontiguousarray(t.reshape(ny, b, nx, b).transpose(0, 2, 1, 3))
    if b > 2 * border + 1:
        inner = tiles[:, :, border:b - border, border:b - border]
    else:
        inner = tiles
    istd = float(inner.reshape(inner.shape[0], inner.shape[1], -1).std(axis=2).mean())
    gx = np.abs(np.diff(t, axis=1))
    gy = np.abs(np.diff(t, axis=0))
    vcols = np.arange(b - 1, t.shape[1] - 1, b)
    vrows = np.arange(b - 1, t.shape[0] - 1, b)
    bj = []
    if len(vcols):
        bj.append(float(gx[:, vcols].mean()))
    if len(vrows):
        bj.append(float(gy[vrows, :].mean()))
    return (float(np.mean(bj)) if bj else 0.0), istd


def _profiles(g, trim=2):
    t = g
    if min(g.shape) > 2 * trim + 8:
        t = g[trim:-trim, trim:-trim]
    col = np.abs(np.diff(t, axis=1)).mean(axis=0)
    row = np.abs(np.diff(t, axis=0)).mean(axis=1)
    return col, row


def score_patch(g, bsizes, jpeg_b=(), border=1, trim=2):
    """对单个 patch 打分：每个块尺寸用最优相位算一次。

    注意：_profiles 会裁掉 trim 个像素再算剖面，所以反解出的相位是在**裁剪后**
    坐标系里的。blockiness 用的是原图，必须把相位加上 trim 补偿回来，
    否则边界整体错位、跳变退化成内部梯度（v2 第一版就栽在这里）。
    """
    best = 1e-6
    bestb = 0
    col, row = _profiles(g, trim)
    off = trim if min(g.shape) > 2 * trim + 8 else 0
    for b in bsizes:
        if b in jpeg_b:
            continue
        # 相位换算有两处偏移必须补：
        #   +off  : 剖面是在裁掉 trim 像素后的坐标系里算的
        #   +1    : 剖面索引 i 对应 |t[:,i+1]-t[:,i]|（差分索引），而 blockiness
        #           的瓦片边界落在差分索引 b-1 处，两者定义相差 1
        py = (best_phase(row, b) + off + 1) % b
        px = (best_phase(col, b) + off + 1) % b
        boundary, istd = blockiness(g, b, py, px, border)
        s = boundary / (istd + 0.005)
        if s > best:
            best = s
            bestb = b
    return best, bestb


def score_image(g, bsizes, jpeg_b=(), grid=3):
    H, W = g.shape
    ph, pw = H // grid, W // grid
    best = 1e-6
    bestb = 0
    for i in range(grid):
        for j in range(grid):
            p = g[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            s, b = score_patch(p, bsizes, jpeg_b)
            if s > best:
                best = s
                bestb = b
    return best, bestb


def pixelate(g, b, frac=0.5):
    H, W = g.shape
    hh, ww = int(H * frac / 2), int(W * frac / 2)
    y0, y1 = H // 2 - hh, H // 2 + hh
    x0, x1 = W // 2 - ww, W // 2 + ww
    reg = g[y0:y1, x0:x1]
    h, w = reg.shape
    hb, wb = h // b, w // b
    reg = reg[:hb * b, :wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3))
    reg = np.repeat(np.repeat(reg, b, axis=0), b, axis=1)
    out = g.copy()
    out[y0:y0 + hb * b, x0:x0 + wb * b] = reg
    return out


def jpeg_roundtrip(g, q=85):
    """把灰度图过一次 JPEG 编解码，模拟真实图片的压缩损失。"""
    import io
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=q)
    b.seek(0)
    return np.asarray(Image.open(b).convert('L'), dtype=np.float32) / 255.0


def validate(data, n, res, bsizes, jpeg_b, seed=0, realistic=True):
    paths = list_images(data)
    random.seed(seed)
    sample = random.sample(paths, min(n, len(paths)))
    clean, mos, pairs = [], [], []
    t0 = time.perf_counter()
    for p in sample:
        try:
            g = load_gray(p, res)
        except Exception:
            continue
        # 两类都过一次 JPEG，保证公平（真实图片本来就是 JPEG）
        gc = jpeg_roundtrip(g) if realistic else g
        clean.append(score_image(gc, bsizes, jpeg_b)[0])
        b = random.choice([6, 10, 14, 20])
        gm = jpeg_roundtrip(pixelate(g, b)) if realistic else pixelate(g, b)
        mos.append(score_image(gm, bsizes, jpeg_b)[0])
        pairs.append((b, clean[-1], mos[-1]))
    dt = time.perf_counter() - t0
    cl, ms = np.array(clean), np.array(mos)
    print('sample %d, %.1fs (%.0f ms/img, 2 scorings each), jpeg_roundtrip=%s' % (
        len(cl), dt, dt / max(1, len(cl)) * 1000 / 2, realistic))
    print('  clean    : median %10.3f  p90 %10.3f  max %12.3f' % (
        np.median(cl), np.percentile(cl, 90), cl.max()))
    print('  mosaicked: median %10.3f  p10 %10.3f  min %12.3f' % (
        np.median(ms), np.percentile(ms, 10), ms.min()))

    labels = np.r_[np.zeros(len(cl)), np.ones(len(ms))]
    sc = np.r_[cl, ms]
    order = np.argsort(sc)
    ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
    n1, n0 = labels.sum(), (1 - labels).sum()
    auc = (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    print('  AUC = %.4f' % auc)

    best = None
    for th in np.unique(np.r_[sc, sc * 1.0001]):
        tp = int((ms >= th).sum()); fn = int((ms < th).sum())
        fp = int((cl >= th).sum())
        rec = tp / max(1, tp + fn); prec = tp / max(1, tp + fp)
        if prec >= 0.90 and (best is None or rec > best[1]):
            best = (th, rec, prec)
    if best:
        print('  at precision>=90%%: threshold %.3f -> recall %.1f%%' % (best[0], best[1] * 100))
    else:
        print('  cannot reach 90% precision with useful recall')

    print('\n  per block size:')
    for b in sorted(set(x[0] for x in pairs)):
        sel = [(c, m) for bb, c, m in pairs if bb == b]
        miss = sum(1 for c, m in sel if m < 50)
        print('    block=%2d  n=%3d  missed(score<50) %d' % (b, len(sel), miss))
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--bmin', type=int, default=4)
    ap.add_argument('--bmax', type=int, default=28)
    ap.add_argument('--grid', type=int, default=3)
    ap.add_argument('--exclude-jpeg', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    bsizes = list(range(a.bmin, a.bmax + 1))
    jpeg_b = (8, 16, 24) if a.exclude_jpeg else ()
    print('blocks %d~%d, exclude %s, grid %dx%d' % (
        a.bmin, a.bmax, list(jpeg_b) or 'none', a.grid, a.grid))

    if a.validate:
        validate(a.data, a.n, a.res, bsizes, jpeg_b)
        return

    paths = list_images(a.data)
    if a.limit:
        paths = paths[:a.limit]
    rows = []
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        try:
            g = load_gray(p, a.res)
            s, b = score_image(g, bsizes, jpeg_b, a.grid)
            rows.append(dict(src=p, score=round(float(s), 4), block=int(b)))
        except Exception as e:
            rows.append(dict(src=p, score=-1, block=0, err=type(e).__name__))
        if (i + 1) % 500 == 0:
            print('  %d/%d %.1fs' % (i + 1, len(paths), time.perf_counter() - t0), flush=True)
    sc = sorted(r['score'] for r in rows)
    n = len(sc)
    if n:
        q = lambda p: sc[min(n - 1, int(p * n))]
        print('\ndistribution: min=%.3f p25=%.3f med=%.3f p75=%.3f p90=%.3f p95=%.3f p99=%.3f max=%.3f' % (
            sc[0], q(.25), q(.5), q(.75), q(.90), q(.95), q(.99), sc[-1]))
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print('-> %s' % a.out)


if __name__ == '__main__':
    main()
