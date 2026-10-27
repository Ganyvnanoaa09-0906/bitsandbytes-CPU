# -*- coding: utf-8 -*-
"""M2 fix #2: scale_add's s is half* -> ls1/ls2 must be uploaded fp16."""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_block.py'
s = io.open(p, encoding='utf-8').read()
old = "    dls1 = up(ls1); dls2 = up(ls2)"
new = ("    # scale_add 的 s 也是 half*，ls=1e-2 以 float32 传进去会变成 ~0 的次正规数，\n"
       "    # 整块的残差贡献直接消失（上次就是这么“通过”的）。\n"
       "    dls1 = up(ls1.astype(np.float16)); dls2 = up(ls2.astype(np.float16))")
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched')
