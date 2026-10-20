"""qwen4bit_loader.py — reusable loader for the Qwen-Image-2.1 4-bit NB4 transformer.

The 4-bit directory is bitsandbytes NF4 prequantized (packed uint8 + quant_meta.json),
which diffusers' from_pretrained cannot consume.  This module:
  * instantiates QwenImage21Transformer2DModel on the meta device,
  * replaces all 232 nn.Linear with bnb Linear4bit carrying the packed weights,
  * loads the non-quantized tensors from the 4-bit shards,
  * returns a CPU float32 model ready for forward passes (~4.2 GB RSS).
"""
from __future__ import annotations

import json

import torch
from accelerate import init_empty_weights
from diffusers.models import QwenImage21Transformer2DModel
from safetensors import safe_open

from bitsandbytes.nn import Linear4bit, Params4bit

Q_DIR = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer'


def _read_quant_layer(name: str, shard_map):
    shard = shard_map[name]
    qs = {}
    with safe_open(Q_DIR + '\\' + shard, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        packed = f.get_tensor(name).clone()
        for suffix in ('absmax', 'nested_absmax', 'nested_quant_map', 'quant_map',
                       'quant_state.bitsandbytes__nf4'):
            k = name + '.' + suffix
            if k in keys:
                qs[suffix] = f.get_tensor(k).clone()
    return packed, qs


def _set_submodule(model, name, new):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def load_transformer(verbose: bool = True) -> QwenImage21Transformer2DModel:
    torch.set_num_threads(6)
    cfg = QwenImage21Transformer2DModel.load_config(Q_DIR)
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(cfg)
    qmeta = json.load(open(Q_DIR + r'\quant_meta.json'))['layers']
    qmap = json.load(open(Q_DIR + r'\diffusion_pytorch_model.safetensors.index.json'))['weight_map']
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    for i, (n, m) in enumerate(linears):
        key = n + '.weight'
        if key not in qmeta:
            continue
        packed, qs = _read_quant_layer(key, qmap)
        lin = Linear4bit(m.in_features, m.out_features, bias=m.bias is not None,
                         compute_dtype=torch.float32, compress_statistics=True,
                         quant_type='nf4')
        lin.weight = Params4bit.from_prequantized(packed, qs, requires_grad=False,
                                                  device='cpu', module=lin)
        _set_submodule(model, n, lin)

    quant_weights = {n + '.weight' for n, _ in linears if n + '.weight' in qmeta}
    prefixes = tuple(quant_weights) + tuple(w + '.' for w in quant_weights)
    wanted = [k for k in model.state_dict()
              if not (k in quant_weights or any(k.startswith(p) for p in prefixes))]
    sd = {}
    by_shard = {}
    for k in wanted:
        if k not in qmap:
            continue
        by_shard.setdefault(qmap[k], []).append(k)
    for shard, keys in by_shard.items():
        with safe_open(Q_DIR + '\\' + shard, framework='pt', device='cpu') as f:
            for k in keys:
                sd[k] = f.get_tensor(k).float()
    model.load_state_dict(sd, strict=False, assign=True)
    model.eval()
    if verbose:
        import psutil
        print('Qwen-Image 4bit transformer loaded | %.2fB params | RSS %.2f GB'
              % (sum(p.numel() for p in model.parameters()) / 1e9,
                 psutil.Process().memory_info().rss / 1e9), flush=True)
    return model
