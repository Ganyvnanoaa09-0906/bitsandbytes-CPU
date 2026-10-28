# -*- coding: utf-8 -*-
"""bench_forge_clean.py —— 趁机器空闲做一次干净测量（一次性跑完）。

必须在【没有别的重负载】时跑。跑之前会检查 CPU 占用并警告。

内容：
  ① 逐形状：torch vs forge(auto=0 固定分块) vs forge(auto=1 自适应)，min-of-N
  ② 线程数扫描（4/5/6）—— 之前量过「6 线程比 5 线程慢 5.9%（15W 功耗墙）」，
     但那是在别的负载下量的，这里复核
  ③ 端到端：整步 A/B（torch / forge-all / forge-ffn），校验 loss+梯度一致
  ④ 按每步调用次数加权，估算「整步里 GEMM 占多少」，并给出
     「如果 GEMM 跑到 X GFLOPS，整步会是多少」的换算

★ 只看墙钟。torch profiler 的 self_cpu_time_total 是跨线程求和的，占比不可用
  （已经用 ab_rmsnorm.py 证伪过一次：profiler 说 26%，实测 2.9%）。
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
import os
import sys
import time
import json

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

from scripts.lib import forge_gemm
from scripts.lib import forge_patch
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CK = os.environ.get('CKPT', r'D:\work\bitsandbytes-CPU\.autopilot\v4_bos\ckpt_13000.pt')
B = int(os.environ.get('B', '8'))
REPS = int(os.environ.get('REPS', '5'))

# (名字, 布局, I, J, R, 每步次数)  —— 次数来自 probe_step_breakdown.py 的实测
SHAPES = [
    ('w12 前向',  'NT', 2048, 2736,  512, 10),
    ('w12 grad_x', 'NN', 2048,  512, 2736, 10),
    ('w12 grad_W', 'TN', 2736,  512, 2048, 10),
    ('w3  前向',  'NT', 2048,  512, 1368, 10),
    ('w3  grad_x', 'NN', 2048, 1368,  512, 10),
    ('w3  grad_W', 'TN',  512, 1368, 2048, 10),
    ('方阵 前向',  'NT', 2048,  512,  512, 20),
    ('方阵 grad_x', 'NN', 2048,  512,  512, 20),
    ('方阵 grad_W', 'TN',  512,  512, 2048, 20),
    ('256 前向',  'NT', 2048,  256,  512, 20),
    ('256 grad_x', 'NN', 2048,  512,  256, 20),
    ('256 grad_W', 'TN',  256,  512, 2048, 20),
    ('head gen',  'NT', 2048, 1024,  512, 1),
    ('head think', 'NT',  128, 1024,  512, 1),
]

print('=' * 100)
print('forge AVX2 内核 · 干净测量')
print('=' * 100)

try:
    import psutil
    cpu = psutil.cpu_percent(interval=1.0)
    vm = psutil.virtual_memory()
    print('  当前 CPU 占用 %.0f%%   内存可用 %.1f/%.1f GB' % (cpu, vm.available / 1e9, vm.total / 1e9))
    if cpu > 25:
        print('  !! 警告：CPU 还有 %.0f%% 占用，测量会被污染' % cpu)
except Exception:
    pass
print()


def mk(shape, layout):
    _, _, I, J, R, _ = shape          # (名字, 布局, I, J, R, 次数) —— 6 个元素
    if layout == 'NT':
        return torch.randn(I, R), torch.randn(J, R)
    if layout == 'NN':
        return torch.randn(I, R), torch.randn(R, J)
    return torch.randn(R, I), torch.randn(R, J)


def torch_mm(A, Bm, lay):
    if lay == 'NT':
        return torch.mm(A, Bm.t())
    if lay == 'NN':
        return torch.mm(A, Bm)
    return torch.mm(A.t(), Bm)


def gflops(I, J, R, dt):
    return 2.0 * I * J * R / dt / 1e9


# ================= ① 逐形状 =================
print('---- ① 逐形状吞吐（GFLOPS，越高越好） ----')
print('  %-11s %-4s %-20s %8s %9s %9s %8s %8s'
      % ('形状', '布局', 'I x J x R', 'GFLOP', 'torch', 'forge固定', 'forge自适', '自适/固定'))
print('  ' + '-' * 94)

results = []
for sh in SHAPES:
    nm, lay, I, J, R, cnt = sh
    A, Bm = mk(sh, lay)
    ref = FG.torch_ref(A, Bm, lay)

    for _ in range(2):
        torch_mm(A, Bm, lay)
    bt = 1e30
    for _ in range(REPS):
        t0 = time.perf_counter()
        torch_mm(A, Bm, lay)
        bt = min(bt, time.perf_counter() - t0)

    g = {}
    for auto in (0, 1):
        FG.set_auto(auto)
        for _ in range(2):
            FG.mm(A, Bm, lay)
        b = 1e30
        for _ in range(REPS):
            t0 = time.perf_counter()
            FG.mm(A, Bm, lay)
            b = min(b, time.perf_counter() - t0)
        g[auto] = gflops(I, J, R, b)
    FG.set_auto(1)

    fl = 2.0 * I * J * R / 1e9
    gt = gflops(I, J, R, bt)
    print('  %-11s %-4s %-20s %8.2f %9.1f %9.1f %8.1f %8.2fx'
          % (nm, lay, '%d x %d x %d' % (I, J, R), fl, gt, g[0], g[1], g[1] / g[0]))
    results.append(dict(name=nm, layout=lay, I=I, J=J, R=R, count=cnt, gflop=fl,
                        torch=gt, forge_fixed=g[0], forge_auto=g[1]))

print()
tot_fl = sum(r['gflop'] * r['count'] for r in results)
print('  每步 GEMM 总 FLOPs: %.2f GFLOP' % tot_fl)
print()
print('  %-14s %12s %12s %12s' % ('版本', 'GEMM 总时间', '整步推算', '相对 torch'))
print('  ' + '-' * 56)
base = None
for tag, key in (('torch', 'torch'), ('forge固定', 'forge_fixed'), ('forge自适应', 'forge_auto')):
    t = sum(r['gflop'] * r['count'] / r[key] for r in results)
    if base is None:
        base = t
    print('  %-14s %10.0f ms %11s %11.3fx'
          % (tag, t * 1e3, '—', base / t))

# ================= ② 线程扫描 =================
print()
print('---- ② 线程数扫描（最大的那个形状 2048x2736x512 NT） ----')
A = torch.randn(2048, 512)
Bm = torch.randn(2736, 512)
print('  %-8s %12s %12s' % ('线程', 'forge GFLOPS', 'torch GFLOPS'))
print('  ' + '-' * 36)
for nt in (3, 4, 5, 6):
    torch.set_num_threads(nt)
    FG.set_threads(nt)
    for _ in range(2):
        torch.mm(A, Bm.t())
        FG.mm(A, Bm, 'NT')
    bt = 1e30
    for _ in range(REPS):
        t0 = time.perf_counter(); torch.mm(A, Bm.t()); bt = min(bt, time.perf_counter() - t0)
    bf = 1e30
    for _ in range(REPS):
        t0 = time.perf_counter(); FG.mm(A, Bm, 'NT'); bf = min(bf, time.perf_counter() - t0)
    f = 2.0 * 2048 * 2736 * 512 / 1e9
    print('  %-8d %12.1f %12.1f' % (nt, f / bf, f / bt))
torch.set_num_threads(6)
FG.set_threads(6)

# ================= ③ 端到端 =================
print()
print('---- ③ 端到端整步 A/B（B=%d） ----' % B)
ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
sd = ck.get('model') or ck.get('state_dict') or ck
T = cfg.total_tokens
torch.manual_seed(1234)
tok = torch.randint(0, cfg.vocab_size, (B, T))


def build():
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(sd)
    m.train()
    return m


def flat(o):
    r = []
    if o is None:
        return r
    if isinstance(o, (tuple, list)):
        for i in o:
            r += flat(i)
        return r
    if torch.is_tensor(o):
        r.append(o)
    return r


def run(m):
    out = m.forward_logits(tok)
    sum(f.sum() for f in flat(out)).backward()


def timeit(m, n=REPS):
    for _ in range(2):
        m.zero_grad(set_to_none=True)
        run(m)
    best = 1e30
    for _ in range(n):
        m.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        run(m)
        best = min(best, time.perf_counter() - t0)
    return best


m0 = build(); m0.zero_grad(set_to_none=True); run(m0)
L0 = sum(f.sum().item() for f in flat(m0.forward_logits(tok)))
G0 = {n: p.grad.detach().clone() for n, p in m0.named_parameters() if p.grad is not None}

step = {}
for tag, only in (('torch', None), ('forge-all', None), ('forge-ffn', ('w12', 'w3'))):
    m = build()
    nrep = 0
    if tag != 'torch':
        nrep = len(forge_patch.enable(m, only=only))
    if tag != 'torch':
        m.zero_grad(set_to_none=True)
        run(m)
        L1 = sum(f.sum().item() for f in flat(m.forward_logits(tok)))
        worst = 0.0
        for n, p in m.named_parameters():
            if p.grad is None or n not in G0:
                continue
            d = (p.grad - G0[n]).norm().item() / max(G0[n].norm().item(), 1e-30)
            worst = max(worst, d)
        print('  %-10s loss 相对差 %.2e  梯度最大相对差 %.2e  (替换 %d 个)' % (
            tag, abs(L1 - L0) / max(abs(L0), 1e-30), worst, nrep))
        m.zero_grad(set_to_none=True)
    t = timeit(m)
    step[tag] = t
    forge_patch.disable(m)
    print('  %-10s %9.1f ms/步' % (tag, t * 1e3))

bt = step['torch']
print()
print('  %-12s %11s %10s %14s' % ('版本', 'ms/步', '加速比', '1000步省(分)'))
print('  ' + '-' * 52)
for tag in ('torch', 'forge-all', 'forge-ffn'):
    print('  %-12s %11.1f %9.3fx %14.1f' % (tag, step[tag] * 1e3, bt / step[tag],
                                            (bt - step[tag]) * 1000 / 60))

out = dict(cpu_free=True, shapes=results, tot_gflop=tot_fl,
           gemm_est={k: sum(r['gflop'] * r['count'] / r[k] for r in results) for k in
                     ('torch', 'forge_fixed', 'forge_auto')},
           step_ms=step)
with open(r'D:\work\bitsandbytes-CPU\.autopilot\forge_clean_bench.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print()
print('已写入 .autopilot\\forge_clean_bench.json')
print('DONE')
