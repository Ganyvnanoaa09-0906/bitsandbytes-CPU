"""Verify token_scale: coordinates stay inside the grid, and scale=1 is unchanged.

The first attempt at this feature scaled coordinates taken from self.order, which is a
permutation over the full grid. Taking its first T entries for a smaller image therefore
still spanned the whole picture -- measured coordinates ran to 62 and 124 against a grid of
32. The fix indexes the smaller grid directly and then scales up, so a 16x16 image uses even
coordinates of the shared table.

Three things to check:
  * every scale produces coordinates inside [0, grid);
  * the tokens covered are exactly the gs*gs of the small grid, with no repeats;
  * scale=1 reproduces the old behaviour exactly, since it must not disturb existing runs.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2  # noqa: E402

ok_all = True
for ts, T in ((1, 1024), (2, 256), (4, 64)):
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=64, n_layer=2, n_head=2, n_kv_head=1,
                             ffn_dim=128, total_tokens=T, grid=32, token_scale=ts)
    m = SmallARImageModelV2(cfg)
    r, c = m.coords(0, T, torch.device('cpu'))
    in_range = int(r.max()) < 32 and int(c.max()) < 32
    # the distinct positions should be exactly the small grid, each once
    pairs = set(zip(r.tolist(), c.tolist()))
    gs = 32 // ts
    expected = gs * gs
    exact = len(pairs) == expected == T
    ok_all = ok_all and in_range and exact
    print('  token_scale=%d  T=%4d  行 %2d..%2d 列 %2d..%2d  不同位置 %d（应 %d）  %s'
          % (ts, T, r.min(), r.max(), c.min(), c.max(), len(pairs), expected,
             '✓' if (in_range and exact) else '✗'))

# scale=1 must be identical to the pre-change behaviour
cfg = SmallImageConfigV2(vocab_size=1024, d_model=64, n_layer=2, n_head=2, n_kv_head=1,
                         ffn_dim=128, total_tokens=1024, grid=32, token_scale=1)
m = SmallARImageModelV2(cfg)
r, c = m.coords(0, 1024, torch.device('cpu'))
g = 32
p = m.order[:1024]
r_old, c_old = (p // g), (p % g)
same = bool((r == r_old.to(r.dtype)).all() and (c == c_old.to(c.dtype)).all())
print()
print('  token_scale=1 与原实现逐位相同: %s' % ('✓' if same else '✗'))
print()
print('全部通过 ✓' if (ok_all and same) else '有失败 ✗')
