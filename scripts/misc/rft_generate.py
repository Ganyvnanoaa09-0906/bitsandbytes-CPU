# -*- coding: utf-8 -*-
"""rft_generate.py — RFT 第一步：从当前 AR 采样，用【高通判别器】挑出纹理最像真的那些。

为什么用 D 而不用统计量（实测依据，见报告 §10.31）：
  · 按"高频占比"挑 → 挑到的是【噪声】（最高样本 0.219 vs 真实 0.048，高出 4.5 倍）
  · 按"对比度"挑   → 挑出来的仍然平滑，高频反而更低
  · 按"高通 D"挑   → 挑出的是【颜料/笔触般的纹理】，且保留率高达 23%（≈每 4 张留 1 张）
  ⇒ D 问的是"纹理像不像真的"，对噪声免疫；统计量只会被噪声骗到。
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
from scripts.lib.train_vqvae_gan import PatchDiscriminator


def load_ar(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, cfg


def highpass(x):
    lp = F.interpolate(F.avg_pool2d(x, 4), size=x.shape[-2:], mode='nearest')
    r = x - lp
    return r / r.std().clamp(min=1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--disc', required=True, help='train_hp_disc.py 存的高通判别器')
    ap.add_argument('--out', required=True, help='挑出的 token 文件 (.pt)')
    ap.add_argument('--n', type=int, default=2048, help='采多少张')
    ap.add_argument('--keep', type=int, default=512, help='留多少张')
    ap.add_argument('--gen-batch', type=int, default=16)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--add-real', type=int, default=512,
                    help='额外掺入多少张真实训练 token（防微调塌缩）')
    ap.add_argument('--real-tokens', default='', help='真实 token 文件（配 --add-real 用）')
    ap.add_argument('--real-upto', type=int, default=27658, help='只用前 N 行真实 token（排除验证集）')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model, cfg = load_ar(a.ar)
    T = cfg.total_tokens
    side = int(round(T ** 0.5))
    vq, _ = vqvae_from_ckpt(a.vqvae)
    dk = torch.load(a.disc, map_location='cpu', weights_only=False)
    D = PatchDiscriminator(in_ch=dk.get('in_ch', 3), ndf=dk['ndf'], n_layers=3)
    D.load_state_dict(dk['state'])
    D.eval()
    dsize = dk.get('size', 128)
    print('[rft] AR step=%s  T=%d  grid=%d  D(size=%d, acc=%.3f)'
          % (torch.load(a.ar, map_location='cpu', weights_only=False).get('step'),
             T, side, dsize, dk.get('acc', -1)), flush=True)

    # ---- 采样 ----
    toks, t0 = [], time.perf_counter()
    done = 0
    while done < a.n:
        nb = min(a.gen_batch, a.n - done)
        with torch.no_grad():
            tk = model.generate(None, steps=T, top_k=a.top_k, temperature=1.0,
                                seed=a.seed + done, batch=nb)
        toks.append(tk)
        done += nb
        if done % (a.gen_batch * 8) == 0 or done >= a.n:
            print('[rft]   采样 %d/%d  (%.0fs, %.2f s/张)'
                  % (done, a.n, time.perf_counter() - t0,
                     (time.perf_counter() - t0) / done), flush=True)
    toks = torch.cat(toks, 0)

    # ---- 解码 + 打分 ----
    scores = []
    for s in range(0, toks.shape[0], 64):
        idx = toks[s:s + 64, :side * side].view(-1, side, side)
        with torch.no_grad():
            emb = vq.vq.codebook(idx)
            z = emb.permute(0, 3, 1, 2).contiguous()
            img = vq.dec(z)
            if img.shape[-1] != dsize:
                img = F.interpolate(img, size=dsize, mode='bilinear', align_corners=False)
            sc = D(highpass(img)).mean(dim=(1, 2, 3))
        scores.append(sc)
    scores = torch.cat(scores, 0)
    print('[rft] D 分数: min %.3f  中位 %.3f  max %.3f'
          % (scores.min(), scores.median(), scores.max()), flush=True)

    order = scores.argsort(descending=True)
    kept = toks[order[:a.keep]].long()
    print('[rft] 保留 top %d/%d（分数 %.3f .. %.3f）'
          % (a.keep, a.n, scores[order[0]], scores[order[a.keep - 1]]), flush=True)

    # ---- 掺入真实 token 防塌缩 ----
    parts = [kept]
    if a.add_real > 0 and a.real_tokens:
        rt = torch.load(a.real_tokens, map_location='cpu', weights_only=False).long()
        rt = rt[:min(a.real_upto, rt.shape[0])]
        g = torch.Generator().manual_seed(a.seed)
        sel = torch.randperm(rt.shape[0], generator=g)[:a.add_real]
        parts.append(rt[sel])
        print('[rft] 掺入真实 token %d 张（从前 %d 行里选，已排除验证区间 %d..）'
              % (sel.numel(), a.real_upto, a.real_upto), flush=True)

    out = torch.cat(parts, 0)
    torch.save(out, a.out)
    torch.save(dict(scores=scores, order=order, kept=a.keep, n=a.n),
               a.out.replace('.pt', '.scores.pt'))
    print('[rft] -> %s  %s  (%.1f MB)' % (a.out, tuple(out.shape), os.path.getsize(a.out) / 1e6))
    print('[rft] 下一步：--resume 当前 ckpt，用这个 token 文件低 LR 微调')


if __name__ == '__main__':
    main()
