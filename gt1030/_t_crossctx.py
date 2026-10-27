# -*- coding: utf-8 -*-
"""探针 36：多 context 下，建完 N 个 block 之后，回头检查 block0 的权重和输出是否还正确。
   判断是『后建的 context 污染了先建的』还是『计算本身错』。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_multi as MM

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck['model']
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
M, MD, WROW = MM.M, MM.MD, MM.WROW

# block0 的参考权重
Wqkv_ref = torch.cat([sd['blocks.0.attn.q.weight'], sd['blocks.0.attn.k.weight'],
                      sd['blocks.0.attn.v.weight']], 0).T.contiguous().numpy().astype(np.float16)


def chk(tag, blk):
    w = blk.dev.down(blk.W['Wqkv'].w, (MD, WROW), np.float16)
    ok = np.array_equal(w, Wqkv_ref)
    x = np.ascontiguousarray((np.random.default_rng(0).standard_normal((M, MD)) * 0.5)
                             .astype(np.float16))
    out = blk.forward(x)
    a = out.astype(np.float32)
    print('%-22s weights_ok=%-5s out_absmax=%.4f mean=%.6f' % (tag, ok, np.abs(a).max(), a.mean()))


keep = []
b0 = MM.Block(sd, 0)
keep.append(b0)
chk('after 1 block', b0)
for i in (1, 2, 3, 5, 9):
    while len(keep) <= i:
        k = len(keep)
        keep.append(MM.Block(sd, k))
    chk('after %d blocks' % (len(keep)), b0)
