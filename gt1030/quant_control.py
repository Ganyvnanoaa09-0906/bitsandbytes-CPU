"""Control int4 quantization vs checkpoint dequant on one layer (GPTQ/AWQ).

Reports weight rel and single-layer output rel for:
  control_sym_g128 : fp16 W -> symmetric int4 g128 -> dequant
  control_asym_g128: fp16 W -> asymmetric int4 g128 -> dequant
  checkpoint      : qweight/qzeros/scales dequant (variant selected)
"""
from __future__ import annotations

import argparse
import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file

BASE = r'D:\work\models\qwen05_base.safetensors'
CKPT = {'gptq': r'D:\work\models\qwen05_gptq.safetensors',
        'awq': r'D:\work\models\qwen05_awq.safetensors'}
PREF = 'model.layers.0.self_attn.q_proj.'
GROUP = 128
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-9))


def q_sym(w, g=GROUP):
    out, inn = w.shape
    x = w.reshape(out, inn // g, g)
    amax = x.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
    scale = amax / 7.0
    q = (x / scale).round().add(8).clamp(0, 15)
    return ((q - 8) * scale).reshape(out, inn)


def q_asym(w, g=GROUP):
    out, inn = w.shape
    x = w.reshape(out, inn // g, g)
    lo = x.amin(dim=2, keepdim=True)
    hi = x.amax(dim=2, keepdim=True)
    scale = ((hi - lo) / 15.0).clamp_min(1e-8)
    zero = (-lo / scale).round().clamp(0, 15)
    q = (x / scale + zero).round().clamp(0, 15)
    return ((q - zero) * scale).reshape(out, inn)


def deq_gptq(qw, qz, sc):
    in8, out = qw.shape
    inn = in8 * 8
    q = torch.empty((inn, out), dtype=torch.int32)
    for k in range(8):
        q[k::8, :] = (qw >> (4 * k)) & 0xF
    ng, _ = qz.shape
    z = torch.empty((ng, out), dtype=torch.int32)
    for k in range(8):
        z[:, k::8] = (qz >> (4 * k)) & 0xF
    g = (torch.arange(inn) // GROUP).clamp(max=ng - 1)
    return ((q - (z[g] + 1)).float() * sc[g].float()).t()


def deq_awq(qw, qz, sc, reverse=True):
    shifts = torch.arange(0, 32, 4)
    iw = torch.bitwise_right_shift(qw[:, :, None], shifts[None, None, :]).to(torch.int8).reshape(qw.shape[0], -1)
    iz = torch.bitwise_right_shift(qz[:, :, None], shifts[None, None, :]).to(torch.int8).reshape(qz.shape[0], -1)
    if reverse:
        rev = torch.arange(iw.shape[-1], dtype=torch.int32).reshape(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
        iw = iw[:, rev]
        iz = iz[:, rev]
    iw = torch.bitwise_and(iw, 0xF)
    iz = torch.bitwise_and(iz, 0xF)
    sc2 = sc.repeat_interleave(GROUP, dim=0)
    iz2 = iz.repeat_interleave(GROUP, dim=0)
    return ((iw - iz2) * sc2).t().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', required=True, choices=['gptq', 'awq'])
    args = ap.parse_args()
    torch.manual_seed(0)
    base = load_file(BASE)
    W = base[PREF + 'weight'].float()
    x = torch.randn(1, W.shape[1]) * 0.5
    ref = x @ W.t()
    rows = []
    for name, Wq in [('control_sym', q_sym(W)), ('control_asym', q_asym(W))]:
        rows.append(dict(name=name, wrel=rel(Wq, W), yrel=rel(x @ Wq.t(), ref)))
    with safe_open(CKPT[args.format], framework='pt', device='cpu') as f:
        qw = f.get_tensor(PREF + 'qweight')
        qz = f.get_tensor(PREF + 'qzeros')
        sc = f.get_tensor(PREF + 'scales')
    if args.format == 'gptq':
        Wq = deq_gptq(qw, qz, sc)
        rows.append(dict(name='checkpoint_gptq', wrel=rel(Wq, W), yrel=rel(x @ Wq.t(), ref)))
    else:
        for rev in (True, False):
            Wq = deq_awq(qw, qz, sc, reverse=rev)
            rows.append(dict(name='checkpoint_awq_rev%s' % int(rev), wrel=rel(Wq, W),
                             yrel=rel(x @ Wq.t(), ref)))
    print('RESULT ' + json.dumps(rows), flush=True)
    for r in rows:
        print('%-24s wrel %.4f yrel %.4f' % (r['name'], r['wrel'], r['yrel']), flush=True)


if __name__ == '__main__':
    main()
