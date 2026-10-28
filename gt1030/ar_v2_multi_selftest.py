# -*- coding: utf-8 -*-
"""F1-M4 自测（多 context 版）：loss / 梯度 / 10 步轨迹 与 torch fp32 对拍。

    python gt1030/ar_v2_multi_selftest.py [--ckpt path] [--steps N] [--lr 5e-5]
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
import argparse
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import torch.nn.functional as F

import gt1030.ar_v2_multi as MM
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CKPT10 = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
TOKENS = r'F:\weibo_cos\tokens_32x32_weibo.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT10)
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--lr', type=float, default=5e-5)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = SmallImageConfigV2(**ck['cfg'])
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(ck['model'], strict=True)
    P = dict(m.named_parameters())
    tok = torch.load(TOKENS, map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    tokens = np.ascontiguousarray(tok[:MM.B, :cfg.total_tokens].astype(np.int64))
    xt = torch.from_numpy(tokens)

    t0 = time.time()
    model = MM.Model(args.ckpt)
    print('built %d layers (%d contexts) in %.1fs' % (model.n_layer, model.n_layer + 1,
                                                      time.time() - t0))
    t0 = time.time()
    loss_dev = model.forward(tokens)
    t_fwd = time.time() - t0
    with torch.no_grad():
        lg = m.forward_logits(xt, None)[1][0]
    loss_t = float(F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1)))
    print('loss: dev=%.6f torch=%.6f diff=%.2e (fwd %.2fs)'
          % (loss_dev, loss_t, abs(loss_dev - loss_t), t_fwd))

    t0 = time.time()
    model.backward(tokens)
    print('bwd %.2fs' % (time.time() - t0))
    t0 = time.time()
    lg2 = m.forward_logits(xt, None)[1][0]
    lo = F.cross_entropy(lg2[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1))
    lo.backward()
    print('torch bwd %.2fs' % (time.time() - t0))

    SC = MM.CE_SCALE

    def cmp(name, a, b):
        a = np.asarray(a, np.float32) / SC; b = np.asarray(b, np.float32)
        if a.shape != b.shape:
            print('%-16s SHAPE %s vs %s' % (name, a.shape, b.shape)); return
        rel = float(np.abs(a - b).max()) / (float(np.abs(b).max()) + 1e-12)
        print('%-16s rel=%.3e %s' % (name, rel, 'OK' if rel < 5e-2 else 'FAIL'))

    print('--- 权重梯度（device fp16 vs torch fp32）---')
    inds = sorted(set([0, model.n_layer // 2, model.n_layer - 1]))
    for li in inds:
        blk = model.blocks[li]
        p = 'blocks.%d.' % li
        W12r = P[p + 'mlp.w12.weight'].grad.T.contiguous().numpy()
        W3r = P[p + 'mlp.w3.weight'].grad.T.contiguous().numpy()
        Wor = P[p + 'attn.o.weight'].grad.T.contiguous().numpy()
        gq = P[p + 'attn.q.weight'].grad.T.contiguous().numpy()
        gk = P[p + 'attn.k.weight'].grad.T.contiguous().numpy()
        gv = P[p + 'attn.v.weight'].grad.T.contiguous().numpy()
        cmp('L%d dW3' % li, blk.dev.down(blk.W['W3'].grad, (MM.FF, MM.MD), np.float16), W3r)
        cmp('L%d dW12' % li, blk.dev.down(blk.W['W12'].grad, (MM.MD, 2 * MM.FF), np.float16), W12r)
        cmp('L%d dWo' % li, blk.dev.down(blk.W['Wo'].grad, (MM.MD, MM.MD), np.float16), Wor)
        cmp('L%d dWqkv' % li, blk.dev.down(blk.W['Wqkv'].grad, (MM.MD, MM.WROW), np.float16),
            np.concatenate([gq, gk, gv], axis=1))
    print('--- 小向量梯度（ln / qk-norm / ls）---')
    for li in inds:
        blk = model.blocks[li]
        p = 'blocks.%d.' % li
        for key, sd_name in (('ln1', 'ln1.w'), ('ln2', 'ln2.w'), ('qw', 'attn.q_norm.w'),
                             ('kw', 'attn.k_norm.w'), ('ls1', 'ls1'), ('ls2', 'ls2')):
            cmp('L%d d%s' % (li, key), blk.dev.down(blk.dw[key], (blk.host[key].size,), np.float32),
                P[p + sd_name].grad.numpy())
    cmp('head dW', model.head.dev.down(model.head.W.grad, (MM.MD, model.head.V), np.float16),
        P['gen_head.lin.weight'].grad.T.contiguous().numpy())
    cmp('dtoken_emb', model.dtok, P['token_emb.weight'].grad.numpy())
    cmp('drow_emb', model.drow, P['row_emb'].grad.numpy())
    cmp('dcol_emb', model.dcol, P['col_emb'].grad.numpy())

    if args.steps <= 0:
        return

    print('\n--- %d 步轨迹（AdamW lr=%.1e clip=1.0）---' % (args.steps, args.lr))
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    ld, lt = [], []
    t0 = time.time()
    for st in range(1, args.steps + 1):
        ts = time.time()
        l = model.forward(tokens)
        model.backward(tokens)
        model.opt_step(args.lr, st, wd=0.01)
        ld.append(l)
        opt.zero_grad(set_to_none=True)
        lgo = m.forward_logits(xt, None)[1][0]
        l2 = F.cross_entropy(lgo[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1))
        l2.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        lt.append(float(l2))
        print('  step %2d dev=%.6f torch=%.6f diff=%+.2e (%.2fs)'
              % (st, l, lt[-1], l - lt[-1], time.time() - ts))
    print('%d 步总计 %.1fs  |diff| max=%.2e mean=%.2e (验收 <0.02)'
          % (args.steps, time.time() - t0, max(abs(a - b) for a, b in zip(ld, lt)),
             float(np.mean([abs(a - b) for a, b in zip(ld, lt)]))))


if __name__ == '__main__':
    import os
    _n = int(os.environ.get('AR_REEXEC', '0'))
    try:
        main()
    except RuntimeError as e:
        msg = str(e)
        if ('upload verify' not in msg and 'OUT_OF_HOST_MEMORY' not in msg) or _n >= 8:
            raise
        print('[reexec] 构建撞上驱动问题（%s），重启进程重试（第 %d 次）' % (msg[:60], _n + 1),
              flush=True)
        os.environ['AR_REEXEC'] = str(_n + 1)
        os.execv(sys.executable, [sys.executable] + sys.argv)
