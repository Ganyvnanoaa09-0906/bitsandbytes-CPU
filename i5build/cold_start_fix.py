"""Cold-start metric + the free lever (loop_L), in one script.

Step 1 of "how do we fix cold start": build a RULER before tuning. Otherwise
every comparison is eyeballing 4 thumbnails, which cannot separate GRN-16-step
from GRN-32-step.

The ruler (no external models, all cheap):
   color_std    per-channel std of pixels; a blank/black image scores ~0
   edge_density mean |gradient|; mush and pure color fields score low
   entropy      histogram entropy of luminance; "one flat colour" scores low
   pair_dist    mean pairwise L2 between samples; catches mode collapse
   near_const   fraction of samples that are essentially one colour

Calibrate the ruler on cases where we ALREADY know the answer:
   GOOD = the 75%-prefix continuations (clear subjects were visible)
   BAD  = the naive cold start (blank / grey mush / pure black)
If the ruler cannot separate those two, it is useless -- fix it first.

Then use it on the free lever: --loop_L above the training L_max. The sampler's
own help says extra loops may be "free" quality (inference is bandwidth bound).
That lever is untested so far.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
HERE = r'D:\work\bitsandbytes-CPU'
OUT = os.path.join(HERE, 'i5build', 'cold_metric')
AR = os.path.join(HERE, 'i5build', 'loop_ar3', 'ckpt.pt', 'ckpt_last.pt')
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_16x16_cmse.pt'
TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt'
os.makedirs(OUT, exist_ok=True)


def panels(path):
    """Split a contact sheet into square panels.

    Bug fixed: the sheet is PORTRAIT when n=8 (2 cols x 4 rows), so the old
    `cols = round(w/h)` gave 0 and crashed with ZeroDivisionError. Work out the
    grid from the aspect ratio and the known panel count instead.
    """
    im = Image.open(path).convert('RGB')
    w, h = im.size
    for n in (8, 4, 6, 3, 2, 1):
        for cols in range(1, n + 1):
            if n % cols:
                continue
            rows = n // cols
            if w % cols == 0 and h % rows == 0 and w // cols == h // rows:
                pw, ph = w // cols, h // rows
                out = []
                for i in range(n):
                    x, y = (i % cols) * pw, (i // cols) * ph
                    out.append(np.asarray(im.crop((x, y, x + pw, y + ph)),
                                          np.float32) / 255.0)
                return out
    raise ValueError('cannot infer panel grid for %dx%d' % (w, h))


def metrics(imgs):
    """Ruler v2.

    v1 failed calibration: color_std was BACKWARDS (saturated colour blobs score
    high, a real photo of a person scores lower) and entropy barely separated.
    Only edge density worked (1.33x). So v2 leans on structure, not colour:
      edge     mean |gradient|
      hf       high-frequency energy ratio (FFT above 1/8 Nyquist) -- real
               images carry detail; mush and flat fields do not
      lap_var  variance of the Laplacian -- classic blur/focus measure
    Colour stats are kept only as diagnostics, never as the verdict.
    """
    import numpy.fft as fft
    cs, ed, hf, lv, nc = [], [], [], [], 0
    for a in imgs:
        g = a.mean(axis=2)
        cs.append(float(a.std(axis=(0, 1)).mean()))
        ed.append(float(np.abs(np.diff(g, axis=0)).mean() +
                         np.abs(np.diff(g, axis=1)).mean()))
        F = np.abs(fft.fftshift(fft.fft2(g - g.mean())))
        cy, cx = F.shape[0] // 2, F.shape[1] // 2
        r = min(cy, cx) // 8
        hi = F.copy()
        hi[cy - r:cy + r, cx - r:cx + r] = 0
        hf.append(float(hi.sum() / max(F.sum(), 1e-9)))
        lap = (np.diff(g, 2, axis=0)[:, :-2] + np.diff(g, 2, axis=1)[:-2, :])
        lv.append(float(lap.var()))
        if a.std() < 0.02:
            nc += 1
    pd = []
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            pd.append(float(np.abs(imgs[i] - imgs[j]).mean()))
    return dict(color_std=np.mean(cs), edge=np.mean(ed), hf=np.mean(hf),
                lap_var=np.mean(lv),
                pair_dist=np.mean(pd) if pd else 0.0,
                near_const=nc / max(1, len(imgs)))


def run(tag, extra, frac=None):
    out = os.path.join(OUT, tag + '.png')
    if os.path.exists(out):
        return out
    if frac is not None:
        cmd = [sys.executable, 'sample_ar_cond.py', '--ar', AR, '--vqvae', VQ,
               '--tokens', TOK, '--out', out, '--n', '8', '--frac', str(frac),
               '--top-k', '64', '--temperature', '1.0', '--seed', '7',
               '--threads', '6']
    else:
        cmd = [sys.executable, 'sample_ar.py', '--ar', AR, '--vqvae', VQ,
               '--out', out, '--n', '8', '--loop-cache', 'perloop',
               '--top-k', '64', '--temperature', '1.0', '--seed', '7',
               '--threads', '6'] + extra
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if not os.path.exists(out):
        print('  [%s] 生成失败: %s' % (tag, (r.stderr or '')[-200:]))
        return None
    return out


def show(tag, m):
    print('  %-12s edge %.5f  hf %.4f  lap_var %.6f  pair_dist %.4f  '
          'colstd %.3f  near_const %.2f'
          % (tag, m['edge'], m['hf'], m['lap_var'], m['pair_dist'],
             m['color_std'], m['near_const']))


def main():
    print('=' * 74)
    print('第一步：校准尺子（用已知好/坏样本）')
    print('=' * 74)
    rows = []
    for name, kw in (('GOOD 75%', dict(frac=0.75)), ('BAD 0%', dict(frac=0.0))):
        p = run('calib_' + name.split()[0].lower(), [], **kw)
        if p:
            m = metrics(panels(p))
            rows.append((name, m))
            show(name, m)

    if len(rows) == 2:
        g, b = rows[0][1], rows[1][1]
        keys = ('edge', 'hf', 'lap_var')
        print('\n  尺子区分度（GOOD/BAD）: %s'
              % ', '.join('%s %.2fx' % (k, g[k] / max(b[k], 1e-9)) for k in keys))
        ok = all(g[k] > b[k] * 1.15 for k in keys)
        print('  ⇒ %s' % ('尺子可用 ✓' if ok else
                          '⚠️ 区分度不足：这几项都还算不出"主体 vs 抽象"的差别 ✗ '
                          '—— 那需要语义特征（CLIP/检测器），不是手工统计量能做到的'))

    print('\n' + '=' * 74)
    print('第二步：免费杠杆 --loop_L（训练 L_max 约为 3）')
    print('=' * 74)
    for L in (1, 2, 3, 5, 8):
        p = run('loopL%d' % L, ['--loop_L', str(L)])
        if p:
            show('L=%d' % L, metrics(panels(p)))

    print('\n' + '=' * 74)
    print('第三步：GRN + loop_L')
    print('=' * 74)
    for L in (3, 8):
        p = run('grnL%d' % L, ['--grn', '--grn-steps', '16', '--grn-shape',
                               'cosine', '--grn-top-k', '64', '--grn-temp',
                               '1.0', '--loop_L', str(L)])
        if p:
            show('GRN L=%d' % L, metrics(panels(p)))
    print('\n图片都在: %s' % OUT)


if __name__ == '__main__':
    main()
