# -*- coding: utf-8 -*-
"""F1-M5：把 device 上的权重导出成 CPU trainer 同格式的 ckpt（可直接给 sample_ar.py 用）。

device 侧的布局约定（跟 CPU 相反）：
    Wqkv 存 (MD, WROW) —— 行是 in，列是 out，W^T 的转置
    Wo   存 (MD, MD)   —— o.weight.T
    W12  存 (MD, 2FF)  —— w12.weight.T
    W3   存 (FF, MD)   —— w3.weight.T
所以导出要转回来。
"""
from __future__ import annotations
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch


def export_ckpt(model, path, step=0, extra=None):
    """model: ar_v2_multi.Model -> 写一个和 train_ar_v2 完全同格式的 ckpt"""
    import gt1030.ar_v2_multi as MM
    MD, FF, WROW, V = MM.MD, MM.FF, MM.WROW, model.head.V
    sd = {}
    # ---- token / pos embeddings（host 侧表格）----
    sd['token_emb.weight'] = torch.from_numpy(model.tok_tab.astype(np.float32))
    sd['row_emb'] = torch.from_numpy(model.row_tab.astype(np.float32))
    sd['col_emb'] = torch.from_numpy(model.col_tab.astype(np.float32))
    # ---- blocks ----
    for i, blk in enumerate(model.blocks):
        d = blk.dev
        p = 'blocks.%d.' % i
        Wqkv = d.down(blk.W['Wqkv'].w, (MD, WROW), np.float16).astype(np.float32)  # (MD, WROW)
        Wo = d.down(blk.W['Wo'].w, (MD, MD), np.float16).astype(np.float32)
        W12 = d.down(blk.W['W12'].w, (MD, 2 * FF), np.float16).astype(np.float32)
        W3 = d.down(blk.W['W3'].w, (FF, MD), np.float16).astype(np.float32)
        nq = model.cfg['n_head'] * model.cfg['d_model'] // model.cfg['n_head']   # = d_model
        nkv = model.cfg['n_kv_head'] * (model.cfg['d_model'] // model.cfg['n_head'])
        # Wqkv 的列依次是 [q | k | v]
        sd[p + 'attn.q.weight'] = torch.from_numpy(Wqkv[:, :nq].T.copy())
        sd[p + 'attn.k.weight'] = torch.from_numpy(Wqkv[:, nq:nq + nkv].T.copy())
        sd[p + 'attn.v.weight'] = torch.from_numpy(Wqkv[:, nq + nkv:].T.copy())
        sd[p + 'attn.o.weight'] = torch.from_numpy(Wo.T.copy())
        sd[p + 'mlp.w12.weight'] = torch.from_numpy(W12.T.copy())
        sd[p + 'mlp.w3.weight'] = torch.from_numpy(W3.T.copy())
        sd[p + 'ln1.w'] = torch.from_numpy(d.down(blk.ln1, (MD,), np.float16).astype(np.float32))
        sd[p + 'ln2.w'] = torch.from_numpy(d.down(blk.ln2, (MD,), np.float16).astype(np.float32))
        hd = model.cfg['d_model'] // model.cfg['n_head']
        sd[p + 'attn.q_norm.w'] = torch.from_numpy(d.down(blk.qw, (hd,), np.float32))
        sd[p + 'attn.k_norm.w'] = torch.from_numpy(d.down(blk.kw, (hd,), np.float32))
        sd[p + 'ls1'] = torch.from_numpy(d.down(blk.ls1, (MD,), np.float32))
        sd[p + 'ls2'] = torch.from_numpy(d.down(blk.ls2, (MD,), np.float32))
    # ---- head ----
    Wh = model.head.dev.down(model.head.W.w, (MD, V), np.float16).astype(np.float32)  # (MD,V)
    sd['gen_head.lin.weight'] = torch.from_numpy(Wh.T.copy())
    # think_head 在训练里不用（loss 只走 gen_head），保持原 ckpt 的拷贝
    ck_src = getattr(model, 'ck', None)
    if ck_src is not None and 'think_head.lin.weight' in ck_src['model']:
        sd['think_head.lin.weight'] = ck_src['model']['think_head.lin.weight'].clone()
    out = {'model': sd, 'step': int(step), 'cfg': dict(model.cfg)}
    if ck_src is not None:
        for k in ('cluster_of', 'offset_of'):
            if k in ck_src:
                out[k] = ck_src[k]
    if extra:
        out.update(extra)
    torch.save(out, path)
    return path


def ckpt_diff(ck_src, ck_new):
    """对比两个 ckpt 的权重（max_abs / rel）"""
    a, b = ck_src['model'], ck_new['model']
    rows = []
    for k in sorted(a.keys()):
        if k not in b:
            rows.append((k, 'MISSING', 0.0))
            continue
        x, y = a[k].float(), b[k].float()
        if x.shape != y.shape:
            rows.append((k, 'SHAPE %s vs %s' % (tuple(x.shape), tuple(y.shape)), 0.0))
            continue
        err = float((x - y).abs().max())
        rel = err / (float(x.abs().max()) + 1e-12)
        rows.append((k, '%.3e' % err, rel))
    return rows
