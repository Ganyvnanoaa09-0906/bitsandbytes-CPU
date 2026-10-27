# -*- coding: utf-8 -*-
"""One-shot repair of gt1030/ar_v2_block_bwd.py (mojibake comment + abs delta check)."""
import io, re, sys

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_block_bwd.py'
s = io.open(p, encoding='utf-8').read()

start = s.index("    # x2 - x ")
end = s.index("    print('--- bwd ---')")
new = (
    "    # x2 - x 是块的真实贡献；直接比 x2 会被残差 x 盖住（两个残差项各带 ls=1e-2 缩放）。\n"
    "    # 但块贡献只有 1.4e-2 量级，而 x2 以 fp16 存储（|x|~2.5 时 ulp ~1e-3），\n"
    "    # 所以这里只能按【绝对误差】判定，相对误差没有意义。\n"
    "    x2_ref = out.detach().view(M, MD).numpy()\n"
    "    delta_ref = x2_ref - x.astype(np.float32)\n"
    "    de = float(np.abs((x2_np - x.astype(np.float32)) - delta_ref).max())\n"
    "    print('delta    max_abs=%.3e refmax=%.3e %s  (abs-only, fp16 storage floor ~1e-3)'\n"
    "          % (de, float(np.abs(delta_ref).max()), 'OK' if de < 5e-3 else 'FAIL'))\n"
)
s = s[:start] + new + s[end:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)

# report any remaining suspicious characters
bad = [(i + 1, l) for i, l in enumerate(s.splitlines())
       if any('\ue000' <= ch <= '\uf8ff' or ch == '\ufffd' for ch in l)]
print('remaining corrupted lines:', len(bad))
for n, l in bad[:10]:
    print(' ', n, l[:80])
print('OK wrote', p)
