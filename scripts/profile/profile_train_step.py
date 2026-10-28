# -*- coding: utf-8 -*-
"""profile_train_step.py - 拆解 v2 训练单步耗时：前向 / 反向 / 优化器 各占多少。

用法:
  python profile_train_step.py --tokens <base>\\tokens_final2.pt --vqvae <base>\\vqvae_ema_full2\\ckpt_last.pt
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, os, sys, time
import torch
import torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2, build_cluster_map
from scripts.lib.small_vqvae import VQVAE
import bitsandbytes
import psutil


def rss():
    return psutil.Process().memory_info().rss / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--threads', type=int, default=5)
    ap.add_argument('--profile', action='store_true', help='额外跑 torch.profiler 出算子级 top')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True

    tokens = torch.load(a.tokens, map_location='cpu')
    N = tokens.shape[0]
    vk = torch.load(a.vqvae, map_location='cpu', weights_only=False)
    vq = VQVAE(embed_dim=64, n_codes=1024)
    vq.load_state_dict(vk['model'])
    cluster_of, offset_of, per = build_cluster_map(vq.vq.codebook.weight.data, 32)
    n_cluster = int(cluster_of.max().item()) + 1
    del vq

    cfg = SmallImageConfigV2(d_model=512, n_layer=10, n_head=8, n_kv_head=4, ffn_dim=2048,
                             n_cluster=n_cluster, cond_mode='none', head_mode='flat')
    model = SmallARImageModelV2(cfg)
    nparam = sum(p.numel() for p in model.parameters())
    print('params=%.2fM  threads=%d  T=%d  batch=%d' % (nparam / 1e6, a.threads, a.T, a.batch))

    opt = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4)
    B = a.batch

    def tf(x):
        return cluster_of[x], offset_of[x].clamp(max=per - 1)

    def one_step():
        idx = torch.randint(0, N, (B,))
        x = tokens[idx, :a.T].long()
        t0 = time.perf_counter()
        th, ge = model.forward_logits(x, None)
        t1 = time.perf_counter()
        if isinstance(ge, tuple) and ge[1] is not None:
            # factorized head: (cluster_logits, offset_logits)
            gec, ged = ge
            tgt_c, tgt_o = tf(x)
            gc, go = gec[:, :-1], ged[:, :-1]
            loss = (F.cross_entropy(gc.reshape(-1, n_cluster), tgt_c[:, 1:].reshape(-1))
                    + F.cross_entropy(go.reshape(-1, per), tgt_o[:, 1:].reshape(-1)))
        else:
            # flat head: FlatHead.logits 也返回二元组，但第二个是 None
            g0 = ge[0] if isinstance(ge, tuple) else ge
            gc = g0[:, :-1]
            loss = F.cross_entropy(gc.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        t2 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        t3 = time.perf_counter()
        return (t1 - t0), (t2 - t1), (t3 - t2), float(loss)

    print('暖机 %d 步...' % a.warmup)
    for _ in range(a.warmup):
        one_step()

    fw, bw, op, ls = [], [], [], []
    for i in range(a.steps):
        x, y, z, l = one_step()
        fw.append(x); bw.append(y); op.append(z); ls.append(l)

    def stat(v):
        v = sorted(v)
        return v[len(v) // 2]

    mf, mb, mo = stat(fw), stat(bw), stat(op)
    tot = mf + mb + mo
    print()
    print('中位单步耗时拆分:')
    print('  前向     %7.1f ms   %5.1f%%' % (mf * 1000, 100 * mf / tot))
    print('  反向     %7.1f ms   %5.1f%%' % (mb * 1000, 100 * mb / tot))
    print('  优化器   %7.1f ms   %5.1f%%' % (mo * 1000, 100 * mo / tot))
    print('  ------------------------------')
    print('  合计     %7.1f ms' % (tot * 1000))
    print('  loss(末) %.4f   峰值 RSS %.2f GB' % (ls[-1], rss()))

    toks = B * a.T
    print()
    print('吞吐 %.1f tok/s' % (toks / tot))
    # FLOPs 估算：只算线性层（attn + ffn + head），前向 2*params*tokens，反向 2x
    fl = 6 * nparam * toks
    print('估算 %.1f GFLOP/步  =>  %.1f GFLOPS  (含反向 2x)' % (fl / 1e9, fl / tot / 1e9))

    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        print()
        print('torch.profiler 算子级 top 15 ...')
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            for _ in range(3):
                one_step()
        print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=15))


if __name__ == '__main__':
    main()
