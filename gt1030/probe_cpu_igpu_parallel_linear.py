"""probe_cpu_igpu_parallel_linear.py — split one 4-bit linear across CPU and iGPU.

Layer-by-layer there is no parallelism (output of layer i feeds i+1), but a single linear
is independent across rows: rows [0:m] can run on the CPU bnb path while rows [m:M] run on
the iGPU hybrid path in another thread. If the two engines really overlap, wall time can
approach max(cpu_part, gpu_part) instead of cpu_part + gpu_part.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from safetensors import safe_open

from bitsandbytes.nn import Linear4bit, Params4bit

import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen4bit_igpu_linear_fp32 import GPU4BitLinearF32  # noqa: E402

P = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer'
NAME = 'transformer_blocks.3.img_mlp.gate_layer.weight'

idx = json.load(open(P + r'\diffusion_pytorch_model.safetensors.index.json'))
shard = idx['weight_map'][NAME]
qs = {}
with safe_open(P + '\\' + shard, framework='pt', device='cpu') as f:
    keys = set(f.keys())
    packed = f.get_tensor(NAME).clone()
    for suf in ('absmax', 'nested_absmax', 'nested_quant_map', 'quant_map',
                'quant_state.bitsandbytes__nf4'):
        k = NAME + '.' + suf
        if k in keys:
            qs[suf] = f.get_tensor(k).clone()
meta = json.load(open(P + r'\quant_meta.json'))['layers'][NAME]
out_f, in_f = meta['shape']
lin = Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.float32,
                 compress_statistics=True, quant_type='nf4')
lin.weight = Params4bit.from_prequantized(packed, qs, requires_grad=False, device='cpu',
                                          module=lin)
gpu = GPU4BitLinearF32(lin)


def main():
    torch.set_num_threads(3)
    for M in (64, 256):
        x = torch.randn(M, in_f)
        with torch.no_grad():
            lin(x)  # warmup
            t0 = time.perf_counter(); y_cpu = lin(x); cpu_s = time.perf_counter() - t0
            gpu(x)
            t0 = time.perf_counter(); y_gpu = gpu(x); gpu_s = time.perf_counter() - t0
        rel = float((y_gpu - y_cpu).norm() / y_cpu.norm())
        # parallel split: rows [0:m] CPU, [m:M] GPU
        m = M // 2
        x1, x2 = x[:m], x[m:]
        barrier = threading.Barrier(2)

        def cpu_job():
            barrier.wait()
            with torch.no_grad():
                return lin(x1)

        def gpu_job():
            barrier.wait()
            with torch.no_grad():
                return gpu(x2)

        with ThreadPoolExecutor(max_workers=2) as ex:
            # warmup parallel
            f1 = ex.submit(cpu_job); f2 = ex.submit(gpu_job)
            f1.result(); f2.result()
            t0 = time.perf_counter()
            f1 = ex.submit(cpu_job); f2 = ex.submit(gpu_job)
            y1, y2 = f1.result(), f2.result()
            par_s = time.perf_counter() - t0
        y_par = torch.cat([y1, y2], dim=0)
        rel_par = float((y_par - y_cpu).norm() / y_cpu.norm())
        print('M=%3d | CPU %.3f s | GPU %.3f s | CPU+GPU %.3f s | speedup %.2fx '
              '| serial sum %.3f s | rel %.2e'
              % (M, cpu_s, gpu_s, par_s, (cpu_s + gpu_s) / par_s,
                 cpu_s + gpu_s, rel_par), flush=True)


if __name__ == '__main__':
    main()
