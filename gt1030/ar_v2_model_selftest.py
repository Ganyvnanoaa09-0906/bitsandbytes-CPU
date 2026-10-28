# -*- coding: utf-8 -*-
"""F1-M4 自测：整机（10 层）loss / 梯度 / 10 步轨迹 与 torch fp32 对拍。

用法：
    python gt1030/ar_v2_model_selftest.py            # loss + 梯度对拍
    python gt1030/ar_v2_model_selftest.py --steps 10 # 再加 10 步 loss 轨迹对拍
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
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
import numpy
import torch
import torch.nn.functional

import gt1030.ar_v2_model
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
from scripts.lib.small_image_model_v2 import run_blocks

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
TOKENS = r'F:\weibo_cos\tokens_32x32_weibo.pt'


def build_torch(ck):
    cfg = SmallImageConfigV2(**ck['cfg'])
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(ck['model'], strict=True)
    return m, cfg


def torch_forward(m, tokens):
    think, gen = m.forward_logits(tokens, None)
    lg = gen[0] if isinstance(gen, (tuple, list)) else gen
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--ckpt', default=CKPT)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    m, cfg = build_torch(ck)
    N = ck['cfg']['total_tokens']
    tok_all = torch.load(TOKENS, map_location='cpu')
    if isinstance(tok_all, dict):
        tok_all = tok_all.get('tokens', tok_all.get('ids'))
    tok_all = tok_all.numpy() if torch.is_tensor(tok_all) else np.asarray(tok_all)
    tokens = np.ascontiguousarray(tok_all[:MM.B, :N].astype(np.int64))
    print('tokens', tokens.shape, 'grid', cfg.grid, 'V', cfg.vocab_size)

    t0 = time.time()
    dev = MM.V2ModelDev(args.ckpt)
    print('device model built in %.1fs  (B=%d T=%d M=%d V=%d layers=%d)'
          % (time.time() - t0, MM.B, MM.T, MM.M, dev.V, len(dev.layers)))

    # ---------- 前向 loss ----------
    t0 = time.time()
    dev.forward(tokens)
    loss_dev, inv_cnt = dev.loss_and_dlogits(tokens)
    t_fwd = time.time() - t0

    xt = torch.from_numpy(tokens)
    lg = torch_forward(m, xt)
    src, tgt = lg[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1)
    loss_t = float(F.cross_entropy(src, tgt))
    print('loss: dev=%.6f  torch=%.6f  diff=%.2e  (fwd %.2fs)'
          % (loss_dev, loss_t, abs(loss_dev - loss_t), t_fwd))

    # ---------- 反向梯度 ----------
    t0 = time.time()
    dev.backward(inv_cnt)
    dtok, drow, dcol = dev.embed_backward(tokens)
    t_bwd = time.time() - t0
    t0 = time.time()
    loss_t2 = F.cross_entropy(src, tgt)
    loss_t2.backward()
    print('torch bwd %.2fs   device bwd %.2fs' % (time.time() - t0, t_bwd))

    def dn(p, shape):
        return dev.dev.down(p, shape).astype(np.float32)

    def cmp(name, a, b):
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        if a.shape != b.shape:
            print('%-26s SHAPE %s vs %s' % (name, a.shape, b.shape)); return
        err = float(np.abs(a - b).max()); rel = err / (float(np.abs(b).max()) + 1e-12)
        print('%-26s max_abs=%.3e rel=%.3e refmax=%.3e %s'
              % (name, err, rel, float(np.abs(b).max()), 'OK' if rel < 5e-2 else 'FAIL'))

    P = dict(m.named_parameters())
    print('--- 权重梯度（device fp16 vs torch fp32）---')
    for li in (0, 5, 9):
        ly = dev.layers[li]
        p = 'blocks.%d.' % li
        cmp('L%d dW3' % li, dn(ly.W['W3'].grad, (MM.FF, MM.MD)),
            P[p + 'mlp.w3.weight'].grad.T.numpy())
        cmp('L%d dW12' % li, dn(ly.W['W12'].grad, (MM.MD, 2 * MM.FF)),
            P[p + 'mlp.w12.weight'].grad.T.numpy())
        cmp('L%d dWo' % li, dn(ly.W['Wo'].grad, (MM.MD, MM.MD)),
            P[p + 'attn.o.weight'].grad.T.numpy())
        gq = P[p + 'attn.q.weight'].grad.T.numpy()
        gk = P[p + 'attn.k.weight'].grad.T.numpy()
        gv = P[p + 'attn.v.weight'].grad.T.numpy()
        cmp('L%d dWqkv' % li, dn(ly.W['Wqkv'].grad, (MM.MD, MM.WROW)),
            np.concatenate([gq, gk, gv], axis=1))
    cmp('head dW', dn(dev.head.grad, (MM.MD, dev.V)),
        P['gen_head.lin.weight'].grad.T.numpy())
    print('--- 嵌入梯度 ---')
    cmp('dtoken_emb', dtok, P['token_emb.weight'].grad.numpy())
    cmp('drow_emb', drow, P['row_emb'].grad.numpy())
    cmp('dcol_emb', dcol, P['col_emb'].grad.numpy())

    if args.steps <= 0:
        return

    # ---------- 10 步轨迹 ----------
    print('\n--- %d 步 loss 轨迹（AdamW lr=%.1e, clip=1.0, 两边同参）---' % (args.steps, args.lr))
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)   # 与 CPU 训练同设置
    ld, lt = [], []
    t0 = time.time()
    for st in range(1, args.steps + 1):
        ts = time.time()
        dev.forward(tokens)
        l, ic = dev.loss_and_dlogits(tokens)
        dev.backward(ic)
        dev.embed_backward(tokens)
        dev.adamw_step(args.lr, st, max_norm=1.0)
        ld.append(l)

        opt.zero_grad(set_to_none=True)
        lg = torch_forward(m, xt)
        s2, t2 = lg[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1)
        lo = F.cross_entropy(s2, t2)
        lo.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        lt.append(float(lo))
        print('  step %2d  dev=%.6f  torch=%.6f  diff=%+.2e  (%.2fs)'
              % (st, l, lt[-1], l - lt[-1], time.time() - ts))
    print('10 步总计 %.1fs   |loss 差| max=%.2e  mean=%.2e  (验收 <0.02)'
          % (time.time() - t0, max(abs(a - b) for a, b in zip(ld, lt)),
             float(np.mean([abs(a - b) for a, b in zip(ld, lt)]))))


if __name__ == '__main__':
    import os
    _try = int(os.environ.get('AR_REEXEC', '0'))
    try:
        main()
    except MM.DriverBroken as e:
        if _try >= 5:
            print('!! 连续 %d 次进程级重试仍撞上驱动队列失效，放弃: %s' % (_try, e))
            raise
        print('[reexec] 驱动队列整体失效（%s），重启本进程重试（第 %d 次）'
              % (e, _try + 1), flush=True)
        os.environ['AR_REEXEC'] = str(_try + 1)
        os.execv(sys.executable, [sys.executable] + sys.argv)
