# -*- coding: utf-8 -*-
"""Repair gt1030/ar_v2_block.py (M2): fp16 rms weights + non-degenerate forward check."""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_block.py'
s = io.open(p, encoding='utf-8').read()

old_ln = "    dln1 = up(ln1); dln2 = up(ln2); dqw = up(qw); dkw = up(kw)"
new_ln = ("    # rms_apply / rms_bwd_* 里 w 声明为 half*，必须传 fp16；\n"
          "    # 传 float32 会被解释成 0.0 / 1.875 交替的垃圾半精度数。\n"
          "    dln1 = up(ln1.astype(np.float16)); dln2 = up(ln2.astype(np.float16))\n"
          "    dqw = up(qw); dkw = up(kw)")
assert old_ln in s
s = s.replace(old_ln, new_ln)

old_cmp = """    err = float((torch.from_numpy(out.astype(np.float32)) - ref).abs().max())
    rel = err / float(ref.abs().max())
    print('V2 block fwd: max_abs=%.4e rel=%.4e %s' % (err, rel, 'OK' if rel < 2e-2 else 'FAIL'))"""
new_cmp = """    # 注意：直接比 x2 是【无意义】的 —— x2 = x + ls1*attn + ls2*mlp，
    # x 占了 99.9% 的幅度，ls 又是 1e-2，任何块内错误都会被残差掩盖（第一版就是这么假通过的）。
    # 必须比块贡献 x2 - x；而它只有 1.4e-2 量级、x2 又是 fp16 存储（ulp ~1e-3），
    # 所以只能按绝对误差判定。
    dev = out.astype(np.float32)
    refd = ref.numpy()
    de = float(np.abs((dev - x.astype(np.float32)) - (refd - x.astype(np.float32))).max())
    print('V2 block fwd delta: max_abs=%.4e refmax=%.4e %s (abs-only, fp16 floor ~1e-3)'
          % (de, float(np.abs(refd - x.astype(np.float32)).max()),
             'OK' if de < 5e-3 else 'FAIL'))
    print('V2 block fwd x2   : max_abs=%.4e (residual-dominated, informational only)'
          % float(np.abs(dev - refd).max()))"""
assert old_cmp in s
s = s.replace(old_cmp, new_cmp)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched', p)
