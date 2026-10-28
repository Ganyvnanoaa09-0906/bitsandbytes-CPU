# -*- coding: utf-8 -*-
"""make_gan_data.py — 为诊断判别器造三组图：真实 / 教师强制 / 自由采样。

为什么要三组：
  real vs FREE  → 自由采样产物有多"假"（基准）
  real vs TF    → 教师强制产物有多"假"；**若 D 分辨不出 TF，
                   就从【感知度量】上独立印证了 §10.28 的结论
                   （病在采样，不在每步预测）** —— 频域结论的独立复核
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2


def load_ar(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, cfg


def topk_sample(lg, k, temp=1.0):
    import torch.nn.functional as F
    lg = lg / max(temp, 1e-6)
    v, _ = torch.topk(lg, k, dim=-1)
    thr = v[..., -1, None]
    lg = torch.where(lg < thr, torch.full_like(lg, float('-inf')), lg)
    pr = F.softmax(lg, dim=-1)
    return torch.multinomial(pr.reshape(-1, pr.shape[-1]), 1).reshape(pr.shape[:-1])


def decode(vqvae_path, toks, side):
    vq, _ = vqvae_from_ckpt(vqvae_path)
    idx = toks[:, :side * side].view(-1, side, side)
    with torch.no_grad():
        emb = vq.vq.codebook(idx)
        z = emb.permute(0, 3, 1, 2).contiguous()
        return vq.dec(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--start', type=int, default=27658)
    ap.add_argument('--n', type=int, default=192)
    ap.add_argument('--gen-batch', type=int, default=32)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model, cfg = load_ar(a.ar)
    T = cfg.total_tokens
    side = int(round(T ** 0.5))
    print('[data] T=%d grid=%d n=%d' % (T, side, a.n), flush=True)

    alltok = torch.load(a.tokens, map_location='cpu', weights_only=False).long()
    true = alltok[a.start:a.start + a.n]
    print('[data] true tokens %s' % (tuple(true.shape),), flush=True)

    # 真实
    img_real = decode(a.vqvae, true, side)
    print('[data] real images %s' % (tuple(img_real.shape),), flush=True)

    # 教师强制（一次前向）
    with torch.no_grad():
        _t, gen = model.forward_logits(true, None)
        gen = model.merge(gen[0], gen[1], None) if isinstance(gen, (tuple, list)) else gen
        tf_toks = topk_sample(gen, a.top_k)
    img_tf = decode(a.vqvae, tf_toks, side)
    print('[data] TF images %s  (与真值逐位一致 %.1f%%)'
          % (tuple(img_tf.shape), float((tf_toks == true).float().mean()) * 100), flush=True)

    # 自由采样（分批，避免显存/内存峰值）
    chunks = []
    t0 = time.perf_counter()
    for s in range(0, a.n, a.gen_batch):
        nb = min(a.gen_batch, a.n - s)
        with torch.no_grad():
            tk = model.generate(None, steps=T, top_k=a.top_k, temperature=1.0,
                                seed=a.seed + s, batch=nb)
        chunks.append(tk)
        print('[data]   free %d/%d  (%.0fs)' % (min(s + nb, a.n), a.n,
                                                time.perf_counter() - t0), flush=True)
    free_toks = torch.cat(chunks, 0)
    img_free = decode(a.vqvae, free_toks, side)
    print('[data] FREE images %s' % (tuple(img_free.shape),), flush=True)

    torch.save(dict(real=img_real.half(), tf=img_tf.half(), free=img_free.half(),
                    true_toks=true, tf_toks=tf_toks, free_toks=free_toks,
                    step=int(torch.load(a.ar, map_location='cpu',
                                        weights_only=False).get('step', 0))),
               a.out)
    print('[data] -> %s (%.1f MB)' % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
