"""Probe a dense bf16 LLM, for comparison against the NF4 one.

The NF4 Qwen3-1.7B probe gave: prefill 71% dense GEMM plus 22% weight dequantisation, decode 73.5% in
the fused 4-bit kernel. This is the same kind of model without quantisation, so the differences isolate
what the quantisation costs and what the library work buys.

DeepSeek-Coder-1.3B in bf16, 2.57 GB on disk against the NF4 model's 1.59 GB for 2.03B parameters --
so the quantised model is 38% smaller while holding 56% more parameters. That is the headline the
library exists to produce, and it belongs next to the throughput numbers rather than on its own.

One thing to watch for and report rather than assume: this CPU has AVX2 but no AVX512-BF16, so bf16
matmuls may be emulated and slow. If the numbers look wrong for the parameter count, the dtype is the
first suspect, and the run also measures fp32 so the two can be compared directly.

Same probe shape as before: load, a real forward pass, then prefill and decode profiled by op class.
"""
from __future__ import annotations
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

import os
import statistics
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


MODEL = r'D:\work\models\models\deepseek-coder-1.3b-base'
torch.set_num_threads(6)


def rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float('nan')


print()
print('  === 配置 ===')
import json  # noqa: E402
cfg = json.load(open(os.path.join(MODEL, 'config.json'), encoding='utf-8'))
for k in ('model_type', 'architectures', 'hidden_size', 'num_hidden_layers',
          'num_attention_heads', 'intermediate_size', 'vocab_size', 'torch_dtype',
          'max_position_embeddings'):
    if k in cfg:
        print('    %-22s %s' % (k, str(cfg[k])[:90]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
prompt = 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'
ids = tok(prompt, return_tensors='pt')['input_ids']
print('    prompt %d token' % ids.shape[1])

results = {}
for dtype, dtname in ((torch.bfloat16, 'bf16（原样）'), (torch.float32, 'fp32（对照）')):
    print()
    print('  === %s ===' % dtname)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval()
    lt = time.time() - t0
    print('    加载 %.1f s · RSS %.2f GB' % (lt, rss()))
    npar = sum(p.numel() for p in model.parameters())
    print('    %.2fB 参数 · 权重 %.2f GB' % (npar / 1e9, npar * (2 if dtype == torch.bfloat16 else 4) / 1e9))

    def run(t):
        with torch.no_grad():
            return model(t).logits

    run(ids)
    ts = []
    for _ in range(3):
        t0 = time.time()
        run(ids)
        ts.append(time.time() - t0)
    pf = statistics.median(ts)
    print('    prefill %d token: %.3f s ⇒ %.1f token/s' % (ids.shape[1], pf, ids.shape[1] / pf))

    one = ids[:, :1]
    run(one)
    ts = []
    for _ in range(5):
        t0 = time.time()
        run(one)
        ts.append(time.time() - t0)
    dc = statistics.median(ts)
    print('    decode 1 token: %.3f s ⇒ %.2f token/s' % (dc, 1 / dc))
    results[dtname] = (lt, pf, dc, rss())

    if dtype == torch.bfloat16:
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU]) as pr:
                run(ids)
        ev = [e for e in pr.key_averages() if e.self_cpu_time_total > 0]
        ev.sort(key=lambda e: -e.self_cpu_time_total)
        tot = sum(e.self_cpu_time_total for e in ev)
        print()
        print('    %-46s %12s %8s' % ('算子（prefill）', '自耗时 ms', '占比'))
        print('    ' + '-' * 70)
        for e in ev[:12]:
            print('    %-46s %12.1f %7.1f%%' % (e.key[:46], e.self_cpu_time_total / 1000,
                                                e.self_cpu_time_total / tot * 100))
        pats = {'矩阵乘': ('addmm', 'aten::mm', 'matmul', 'linear'), '注意力': ('attention', 'sdpa', 'bmm'),
                '归一化/激活': ('norm', 'silu', 'gelu', 'softmax'), 'elementwise/搬运':
                ('mul', 'add', 'cat', 'copy', 'to', 'contiguous', 'reshape', 'view', 'permute',
                 'transpose', 'slice', 'select', 'expand', 'narrow', 'fill', 'zero')}
        acc = {k: 0.0 for k in pats}
        oth = 0.0
        for e in ev:
            k = e.key.lower()
            for b, ps in pats.items():
                if any(p in k for p in ps):
                    acc[b] += e.self_cpu_time_total
                    break
            else:
                oth += e.self_cpu_time_total
        print()
        print('    按类别（prefill）:')
        for b, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            print('      %-18s %10.1f ms %7.1f%%' % (b, v / 1000, v / tot * 100))
        print('      %-18s %10.1f ms %7.1f%%' % ('其余', oth / 1000, oth / tot * 100))
        # ★ 关键检查:有没有 dequantize
        deq = [e for e in ev if 'dequant' in e.key.lower()]
        print()
        print('    ★ dequantize 相关算子: %s'
              % ('无' if not deq else ', '.join(e.key for e in deq)))
        print('      ⇒ ★【未量化模型没有反量化开销 —— 而 NF4 那个 prefill 里它占 21.8%】')
    del model

print()
print('  === 与 NF4 Qwen3-1.7B 对照 ===')
print('    %-22s %14s %14s' % ('', 'NF4 Qwen3-1.7B', 'bf16 DS-Coder-1.3B'))
print('    ' + '-' * 54)
print('    %-22s %14s %14s' % ('参数量', '2.03B', '1.30B'))
print('    %-22s %14s %14s' % ('磁盘', '1.59 GB', '2.57 GB'))
for k, v in results.items():
    print('    %-22s %14s %11.1f s' % ('加载 (%s)' % k, '—', v[0]))
print()
print('  ⇒ ★【量化的 2.03B 比未量化的 1.3B 还小 38%,而参数多 56%】')
print('  ⇒ ★【而这条对照同时说明:库侧工作的收益【以内存为主】,因为它已经把这个模型装进来了】')
