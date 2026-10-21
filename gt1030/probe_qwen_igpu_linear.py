"""probe_qwen_igpu_linear.py — can the iGPU beat the CPU on a Qwen-Image 4-bit linear?

Path: read one NF4 linear, dequantize on the CPU to fp16, upload (in,out) weight to the
iGPU, run CLBlast Hgemm, download the output.  Compare against bnb's CPU 4-bit matmul.
This is the deciding measurement for a hybrid CPU-attention / iGPU-linear engine.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open

import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16  # noqa: E402

P = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer'
NAME = 'transformer_blocks.0.attn.to_q.weight'


def load_packed(name):
    idx = json.load(open(P + r'\diffusion_pytorch_model.safetensors.index.json'))
    shard = idx['weight_map'][name]
    qs = {}
    with safe_open(P + '\\' + shard, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        packed = f.get_tensor(name).clone()
        for suffix in ('absmax', 'nested_absmax', 'nested_quant_map', 'quant_map',
                       'quant_state.bitsandbytes__nf4'):
            k = name + '.' + suffix
            if k in keys:
                qs[suffix] = f.get_tensor(k).clone()
    return packed, qs


def main():
    torch.set_num_threads(6)
    packed, qs = load_packed(NAME)
    meta = json.load(open(P + r'\quant_meta.json'))['layers'][NAME]
    out_f, in_f = meta['shape']
    p = Params4bit.from_prequantized(packed, qs, requires_grad=False, device='cpu')
    w_cpu = bnb.functional.dequantize_4bit(p.data, p.quant_state).float()  # (out,in)

    lin = Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.float32,
                     compress_statistics=True, quant_type='nf4')
    lin.weight = Params4bit.from_prequantized(packed, qs, requires_grad=False, device='cpu',
                                              module=lin)
    rt = OpenCLFP16.get()
    print('device', rt.device_name, '| max alloc %.0f MB' % (rt.max_alloc / 1e6))

    for seq in (1, 16, 64, 256):
        x = torch.randn(1, seq, in_f)
        # CPU reference
        with torch.no_grad():
            y_cpu = lin(x)
            t0 = time.perf_counter(); y_cpu = lin(x); cpu_s = time.perf_counter() - t0
        # hybrid CPU-dequant -> GPU GEMM
        t0 = time.perf_counter()
        w_half = w_cpu.numpy().astype(np.float16).T.copy()  # (in,out)
        t_deq = time.perf_counter() - t0
        wb = rt.to_gpu(w_half)
        x_half = x.numpy().reshape(-1, in_f).astype(np.float16)
        xb = rt.to_gpu(x_half)
        t0 = time.perf_counter()
        yb, ysh = rt.linear_h(xb, wb, seq, in_f, out_f)
        gpu_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_np = rt.to_host(yb, ysh)
        down_s = time.perf_counter() - t0
        err = float(np.linalg.norm(y_np.astype(np.float32).reshape(y_cpu.shape) - y_cpu.numpy())
                    / (np.linalg.norm(y_cpu.numpy()) + 1e-9))
        total = t_deq + gpu_s + down_s
        print('seq=%4d | CPU bnb %.3f s | GPU(deq %.3f + gemm %.3f + down %.3f) = %.3f s '
              '| speedup %.2fx | rel %.4f'
              % (seq, cpu_s, t_deq, gpu_s, down_s, total, cpu_s / total, err), flush=True)
        for b in (wb, xb, yb):
            b.release()


if __name__ == '__main__':
    main()
