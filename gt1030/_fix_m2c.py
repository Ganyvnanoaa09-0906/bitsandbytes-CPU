# -*- coding: utf-8 -*-
"""M2 fix #3: scale_add's s is float* after all (revert), and report both sides of the delta."""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_block.py'
s = io.open(p, encoding='utf-8').read()
old = ("    # scale_add 的 s 也是 half*，ls=1e-2 以 float32 传进去会变成 ~0 的次正规数，\n"
       "    # 整块的残差贡献直接消失（上次就是这么“通过”的）。\n"
       "    dls1 = up(ls1.astype(np.float16)); dls2 = up(ls2.astype(np.float16))")
new = "    dls1 = up(ls1); dls2 = up(ls2)   # scale_add 的 s 声明为 float*，这里就是 float32"
assert old in s
s = s.replace(old, new)

old2 = """    de = float(np.abs((dev - x.astype(np.float32)) - (refd - x.astype(np.float32))).max())"""
new2 = """    dev_delta = dev - x.astype(np.float32)
    ref_delta = refd - x.astype(np.float32)
    de = float(np.abs(dev_delta - ref_delta).max())
    print('  [dbg] dev delta max=%.4e  ref delta max=%.4e  cosine=%.5f'
          % (float(np.abs(dev_delta).max()), float(np.abs(ref_delta).max()),
             float((dev_delta * ref_delta).sum() /
                   (np.linalg.norm(dev_delta) * np.linalg.norm(ref_delta) + 1e-30))))"""
assert old2 in s
s = s.replace(old2, new2)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
