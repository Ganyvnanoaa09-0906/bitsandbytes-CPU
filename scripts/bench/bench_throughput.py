# -*- coding: utf-8 -*-
"""bench_throughput.py — 吞吐扫描：找 CPU 上最快的训练配置。

我的 profile 实测：当前配置 97.5 GFLOPS，而 MKL sgemm 在本机能跑到 158~388 GFLOPS
=> 有 1.6~4× 空间，值得系统扫一遍。

扫这些维度（T=256 为主，因为占 96% 的 token）：
  - batch: 8 / 16 / 32
  - threads: 4 / 5 / 6
  - bf16 autocast: 开/关
输出每组的 step_s 与 tok/s，按 tok/s 排序。

用法:
  python bench_throughput.py --tokens <base>\\tokens_final2.pt --vqvae <base>\\vqvae_ema_full2\\ckpt_last.pt
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, itertools, json, os, sys, time, gc
import torch
import torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2, build_cluster_map
from scripts.lib.small_vqvae import VQVAE
import bitsandbytes


def run_cfg(tokens, cluster_of, offset_of, per, n_cluster, threads, batch, T, bf16, warmup, steps):
    torch.set_num_threads(threads)
    cfg = SmallImageConfigV2(d_model=512, n_layer=10, n_head=8, n_kv_head=4, ffn_dim=2048,
                             n_cluster=n_cluster, cond_mode='none', head_mode='flat')
    model = SmallARImageModelV2(cfg)
    model.train()
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4)
    N = tokens.shape[0]

    def tf(x):
        return cluster_of[x], offset_of[x].clamp(max=per - 1)

    def step():
        idx = torch.randint(0, N, (batch,))
        x = tokens[idx, :T].long()
        t0 = time.perf_counter()
        if bf16:
            with torch.autocast('cpu', dtype=torch.bfloat16):
                (thc, thd), (gec, ged) = model.forward_logits(x, None)
                g0 = gec if gec is not None else thd
                g = g0[:, :-1]
                loss = F.cross_entropy(g.reshape(-1, cfg.vocab_size).float(), x[:, 1:].reshape(-1))
        else:
            (thc, thd), (gec, ged) = model.forward_logits(x, None)
            g0 = gec if gec is not None else thd
            g = g0[:, :-1]
            loss = F.cross_entropy(g.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return time.perf_counter() - t0, float(loss.detach())

    try:
        for _ in range(warmup):
            step()
        ts = []
        for _ in range(steps):
            d, l = step()
            ts.append(d)
        ts.sort()
        med = ts[len(ts) // 2]
        toks = batch * T
        return dict(ok=True, step_s=med, tok_s=toks / med,
                    loss=l, rss=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    except Exception as e:
        return dict(ok=False, err='%s: %s' % (type(e).__name__, str(e)[:90]))
    finally:
        del model, opt
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--out', default='bench_throughput.json')
    a = ap.parse_args()

    tokens = torch.load(a.tokens, map_location='cpu')
    vk = torch.load(a.vqvae, map_location='cpu', weights_only=False)
    vq = VQVAE(embed_dim=64, n_codes=1024); vq.load_state_dict(vk['model'])
    cluster_of, offset_of, per = build_cluster_map(vq.vq.codebook.weight.data, 32)
    n_cluster = int(cluster_of.max().item()) + 1
    del vq
    print('tokens=%s  per=%d  T=%d' % (tuple(tokens.shape), per, a.T), flush=True)

    rows = []
    combos = [(8, 5, False), (16, 5, False), (32, 5, False),
              (16, 4, False), (16, 6, False),
              (16, 5, True), (8, 5, True), (32, 6, False)]
    for batch, thr, bf16 in combos:
        tag = 'batch=%-3d threads=%d bf16=%-5s' % (batch, thr, bf16)
        r = run_cfg(tokens, cluster_of, offset_of, per, n_cluster, thr, batch, a.T, bf16,
                    a.warmup, a.steps)
        r.update(batch=batch, threads=thr, bf16=bf16, T=a.T)
        rows.append(r)
        if r['ok']:
            print('  %s  step=%.2fs  %.1f tok/s' % (tag, r['step_s'], r['tok_s']), flush=True)
        else:
            print('  %s  失败: %s' % (tag, r['err']), flush=True)

    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    ok = [r for r in rows if r['ok']]
    ok.sort(key=lambda r: -r['tok_s'])
    print()
    print('================ 按吞吐排序 ================')
    base = None
    for r in ok:
        if r['batch'] == 8 and r['threads'] == 5 and not r['bf16']:
            base = r['tok_s']
    for r in ok:
        gain = (r['tok_s'] / base) if base else 1.0
        print('  %6.1f tok/s  (%.2fx)  batch=%-3d threads=%d bf16=%s'
              % (r['tok_s'], gain, r['batch'], r['threads'], r['bf16']))
    print()
    if base and ok:
        best = ok[0]
        print('  最优: %.1f tok/s  vs 基线 %.1f tok/s  =>  %.2fx'
              % (best['tok_s'], base, best['tok_s'] / base))
    print('  明细 -> %s' % a.out)


if __name__ == '__main__':
    main()
