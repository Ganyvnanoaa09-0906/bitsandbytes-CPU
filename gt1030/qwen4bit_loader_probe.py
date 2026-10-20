"""qwen4bit_loader_probe.py — load the Qwen-Image-2.1 4-bit transformer and run one forward.

The 4-bit directory stores bitsandbytes NF4 prequantized weights plus quant_meta.json.
Diffusers' from_pretrained cannot consume this layout directly, so this script:
  1. instantiates QwenImage21Transformer2DModel on the meta device,
  2. replaces all 232 nn.Linear modules with bnb Linear4bit carrying the packed weights,
  3. loads the non-quantized tensors (norms, embeddings, biases) from the full bf16
     checkpoint (selective safetensors reads),
  4. runs a forward pass on a tiny synthetic latent/text sequence and measures time/RSS.

If this works, the question becomes only whether a full 256px diffusion step is fast
enough; the linear benchmark suggests ~40 s/step at 1024 image tokens on this CPU.
"""
from __future__ import annotations

import json
import time

import torch
from accelerate import init_empty_weights
from diffusers.models import QwenImage21Transformer2DModel
from safetensors import safe_open

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Params4bit

Q_DIR = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer'
FULL_DIR = r'D:\work\qwen_env\Qwen-Image-2.1\transformer'


def read_quant_layer(name: str, shard_map, qmeta):
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


def set_submodule(model, name, new):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def main():
    torch.set_num_threads(6)
    import psutil
    print('meta instantiation...', flush=True)
    cfg = QwenImage21Transformer2DModel.load_config(Q_DIR)
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(cfg)
    print('params %.2fB | meta tensors %d'
          % (sum(p.numel() for p in model.parameters()) / 1e9,
             sum(1 for p in model.parameters() if p.is_meta)), flush=True)

    qmeta = json.load(open(Q_DIR + r'\quant_meta.json'))['layers']
    qmap = json.load(open(Q_DIR + r'\diffusion_pytorch_model.safetensors.index.json'))['weight_map']
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    print('linears', len(linears), flush=True)
    t0 = time.perf_counter()
    for _i, (n, m) in enumerate(linears):
        key = n + '.weight'
        if key not in qmeta:
            print('skip non-quant linear', n)
            continue
        packed, qs = read_quant_layer(key, qmap, qmeta)
        lin = Linear4bit(m.in_features, m.out_features, bias=m.bias is not None,
                         compute_dtype=torch.float32, compress_statistics=True,
                         quant_type='nf4', device='meta')
        lin.weight = Params4bit.from_prequantized(packed, qs, requires_grad=False,
                                                  device='cpu', module=lin)
        set_submodule(model, n, lin)
        if _i % 32 == 0:
            import gc; gc.collect()
            print('  layer %d/%d RSS %.2f GB' % (_i, len(linears), psutil.Process().memory_info().rss/1e9), flush=True)
        if len(globals().get('_done', [])) % 32 == 0:
            pass
    print('replaced linears in %.1f s | RSS %.2f GB'
          % (time.perf_counter() - t0, psutil.Process().memory_info().rss / 1e9), flush=True)

    # non-quant state dict from the full bf16 checkpoint
    full_map = json.load(open(Q_DIR + r'\diffusion_pytorch_model.safetensors.index.json'))['weight_map']
    sd = {}
    quant_weights = {n + '.weight' for n, _ in linears if n + '.weight' in qmeta}
    quant_prefixes = tuple(quant_weights) + tuple(w + '.' for w in quant_weights)
    wanted = [k for k in model.state_dict()
              if not (k in quant_weights or any(k.startswith(p) for p in quant_prefixes))]
    by_shard = {}
    for k in wanted:
        if k not in full_map:
            print('missing in full checkpoint:', k)
            continue
        by_shard.setdefault(full_map[k], []).append(k)
    for shard, keys in by_shard.items():
        with safe_open(Q_DIR + '\\' + shard, framework='pt', device='cpu') as f:
            for k in keys:
                sd[k] = f.get_tensor(k).float()
    print('non-quant tensors', len(sd), flush=True)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print('load_state_dict missing', len(missing), 'unexpected', len(unexpected), flush=True)
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    print('meta params left', len(meta_left), meta_left[:5], flush=True)
    if meta_left:
        raise SystemExit('some parameters stayed on meta; cannot run forward')
    model.eval()

    # tiny synthetic forward: 4x4 latent grid, 8 text tokens
    S, T = 16, 8
    hs = torch.randn(1, S, cfg['in_channels'], dtype=torch.float32)
    enc = torch.randn(1, T, cfg['context_in_dim'], dtype=torch.float32)
    ts = torch.tensor([0.5])
    shapes = [[(1, 4, 4)]]
    mask = torch.cat([torch.zeros(1, T, dtype=torch.bool),
                      torch.ones(1, S // 4, dtype=torch.bool)], dim=1)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(hidden_states=hs, encoder_hidden_states=enc, timestep=ts,
                    img_shapes=shapes, img_mask=mask, return_dict=False)
    dt = time.perf_counter() - t0
    print('FORWARD OK | %.2f s | out %s | RSS %.2f GB'
          % (dt, tuple(out[0].shape), psutil.Process().memory_info().rss / 1e9), flush=True)

    for S2, side in [(64, 8), (256, 16), (1024, 32)]:
        hs2 = torch.randn(1, S2, cfg['in_channels'], dtype=torch.float32)
        shapes2 = [[(1, side, side)]]
        mask2 = torch.cat([torch.zeros(1, T, dtype=torch.bool),
                           torch.ones(1, S2 // 4, dtype=torch.bool)], dim=1)
        t0 = time.perf_counter()
        with torch.no_grad():
            out2 = model(hidden_states=hs2, encoder_hidden_states=enc, timestep=ts,
                         img_shapes=shapes2, img_mask=mask2, return_dict=False)
        dt2 = time.perf_counter() - t0
        print('S=%4d (%dx%d image): %.2f s/forward | out %s | RSS %.2f GB'
              % (S2, side * 16, side * 16, dt2, tuple(out2[0].shape),
                 psutil.Process().memory_info().rss / 1e9), flush=True)


if __name__ == '__main__':
    main()
