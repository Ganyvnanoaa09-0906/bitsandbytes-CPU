# -*- coding: utf-8 -*-
"""test_fuse_qkv.py — 验证 QKV 融合在真实模型上快多少 + 数值是否一致。

用子进程分别构建融合版/分离版，避免同一进程里模块缓存混淆。
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD_PATH = os.path.join(HERE, '_fqkv_child.py')

CHILD = '''
import os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, r"__HERE__")
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

FUSE = os.environ.get("FUSE") == "1"
T, B = 256, 8
torch.set_num_threads(6)
torch.manual_seed(1234)

cfg = SmallImageConfigV2(
    vocab_size=1024, total_tokens=T, grid=16,
    d_model=512, n_layer=10, n_head=8, n_kv_head=4, ffn_dim=2048,
    head_mode="flat", cond_mode="none", pos_mode="2d", token_order="raster",
    loop_times=1, fuse_qkv=FUSE,
)
m = SmallARImageModelV2(cfg)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
x = torch.randint(0, 1024, (B, T))
y = torch.randint(0, 1024, (B, T))


def step():
    out = m.forward_logits(x)
    gen = out[1]
    logits = gen[0] if isinstance(gen, tuple) else gen
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss)


l0 = step()
for _ in range(2):
    step()
t0 = time.perf_counter()
for _ in range(4):
    step()
dt = (time.perf_counter() - t0) / 4
nparam = sum(p.numel() for p in m.parameters())
tag = "FUSED" if FUSE else "SEP"
print(tag, l0, dt, nparam / 1e6)
'''.replace('__HERE__', HERE.replace('\\', '\\\\'))

with open(CHILD_PATH, 'w', encoding='utf-8') as f:
    f.write(CHILD)

res = {}
for fuse in ('0', '1'):
    env = dict(os.environ, FUSE=fuse, PYTHONIOENCODING='utf-8')
    out = subprocess.run([sys.executable, '-u', CHILD_PATH],
                         capture_output=True, text=True, env=env, timeout=1200)
    lines = [l for l in out.stdout.splitlines() if l.startswith(('FUSED', 'SEP'))]
    if not lines:
        print('子进程失败 (FUSE=%s)' % fuse)
        print('--- stdout ---'); print(out.stdout[-1200:])
        print('--- stderr ---'); print(out.stderr[-1200:])
        sys.exit(1)
    tag, loss, dt, np_ = lines[-1].split()
    res[tag] = (float(loss), float(dt), np_)

print('=' * 58)
print('QKV 融合：真实模型 A/B (b=8, T=256, threads=6)')
print('=' * 58)
for k in ('SEP', 'FUSED'):
    if k in res:
        v = res[k]
        print('  %-6s step1_loss=%.4f   单步 %.3f s   %s 参数' % (k, v[0], v[1], v[2]))
if 'SEP' in res and 'FUSED' in res:
    a, b = res['SEP'], res['FUSED']
    print()
    print('  数值  loss 差 %.2e  ->  %s'
          % (abs(a[0] - b[0]), '一致' if abs(a[0] - b[0]) < 1e-3 else '不一致 ⚠️'))
    print('  速度  %.1f%% %s' % (abs(a[1] - b[1]) / a[1] * 100, '更快' if b[1] < a[1] else '更慢'))
    print('  参数  %s -> %s' % (a[2], b[2]))
