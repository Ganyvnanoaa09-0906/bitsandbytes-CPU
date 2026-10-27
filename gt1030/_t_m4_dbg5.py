# -*- coding: utf-8 -*-
"""M4 debug 5: bisect why building the torch model first makes device buffers zero."""
import argparse, sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM

ap = argparse.ArgumentParser()
ap.add_argument('--torchmodel', type=int, default=0)
ap.add_argument('--torchimport', type=int, default=1)
args = ap.parse_args()

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
ck = torch.load(ckpt, map_location='cpu', weights_only=False)
if args.torchmodel:
    from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    m = SmallARImageModelV2(SmallImageConfigV2(**ck['cfg']))
    m.load_state_dict(ck['model'], strict=True)
    print('torch model built, params %.2fM'
          % (sum(p.numel() for p in m.parameters()) / 1e6))

# 设备内存信息
import pyopencl as cl
dev0 = MM.A.ctx.devices[0]
print('device: %s  global_mem=%.2fGB  max_alloc=%.2fGB  local=%.1fKB  cu=%d'
      % (dev0.name, dev0.global_mem_size / 1e9, dev0.max_mem_alloc_size / 1e9,
         dev0.local_mem_size / 1e3, dev0.max_compute_units))
import psutil
vm = psutil.virtual_memory()
print('host RAM: total=%.1fGB avail=%.1fGB' % (vm.total / 1e9, vm.available / 1e9))

model = MM.V2ModelDev(ckpt)
p = model.layers[0].W['Wqkv']
print('W absmax=%.4f' % np.abs(model.dev.down(p.w, (MM.MD, MM.WROW)).astype(np.float32)).max())
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
model.forward(tokens)
print('logits absmax=%.4f'
      % np.abs(model.dev.down(model.sc['logits'], (MM.M, model.V), np.float16).astype(np.float32)).max())
