# -*- coding: utf-8 -*-
"""vq_kmeans_ceiling.py — 便宜的判决实验：码本大小到底值多少 dB？

问题（2026-09-19）：
  全部 16 个已训分词器的码本都是 n_codes=1024（10 bit）—— "加大码本"这条路【从未测过】。
  而 P 的重建是【平滑的糊】，这正是"值量化太粗"的典型症状
  （每个 16x16 块被换成 1024 个原型里最近的那个 ⇒ 块内细节被抹平成均值）。
  实测斜率只有【空间 token 数】的证据（64->256 token 给 +3.5 dB），
  【每 token 的 bit 数】这条路是空的。

做法（不重训 VQ，只重拟合码本）：
  1) 取 P 的训练好 enc/dec；
  2) 在【训练图】上编码出连续 latent z，用 k-means 拟合 K = 1024 / 2048 / 8192 个中心；
  3) 把码本换成这些中心，在【留出图】上编码->解码，量 PSNR / 量化误差。
  ⚠️ 这是【下界】：decoder 是按 1024 码本训的，没为更大码本适配过。
     若这样都能看到明显增益，那么正经重训 8192 码本只会更好。

用法:
  python vq_kmeans_ceiling.py --ckpt <P.pt> --fit 400 --eval 96 --threads 4
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
import argparse, json, os, sys, math, time
import torch
import torch.nn.functional
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt

SIZE = 256
TF = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])


def psnr(a, b):
    return 10.0 * math.log10(4.0 / max(float(F.mse_loss(a, b)), 1e-12))


def kmeans(z, K, iters=25, seed=0):
    """极简 mini-batch 无偏 k-means（余弦空间，因为 VQ 查表时对码本做了 L2 归一化）。"""
    g = torch.Generator().manual_seed(seed)
    zn = F.normalize(z, dim=1)
    perm = torch.randperm(z.shape[0], generator=g)[:K]
    C = z[perm].clone()
    if C.shape[0] < K:                       # 样本不足时补随机
        extra = torch.randint(0, z.shape[0], (K - C.shape[0],), generator=g)
        C = torch.cat([C, z[extra]], 0)
    B = 16384
    for it in range(iters):
        Cn = F.normalize(C, dim=1)
        assign = torch.empty(z.shape[0], dtype=torch.long)
        for s in range(0, z.shape[0], B):
            assign[s:s + B] = (zn[s:s + B] @ Cn.T).argmax(1)
        newC = torch.zeros_like(C)
        cnt = torch.zeros(K)
        newC.index_add_(0, assign, z)
        cnt.index_add_(0, assign, torch.ones(z.shape[0]))
        nonempty = cnt > 0
        newC[nonempty] /= cnt[nonempty].unsqueeze(1)
        C = torch.where(nonempty.unsqueeze(1), newC, C)   # 空簇保留旧中心
    return C, int((~nonempty).sum()) if it == iters - 1 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=r'D:\work\cloud_results\P_attn_26040.pt')
    ap.add_argument('--paths', default=r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt.paths.json')
    ap.add_argument('--fit', type=int, default=400, help='拟合码本用的训练图数')
    ap.add_argument('--eval', type=int, default=96, help='评估用的留出图数')
    ap.add_argument('--ks', default='1024,2048,8192')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    paths = json.load(open(a.paths, encoding='utf-8'))
    # 用训练区间的前段做拟合、后段做留出，避免用验证集（保持验证集干净）
    fit_p = paths[0:a.fit]
    ev_p = paths[8000:8000 + a.eval]

    def load(ps):
        xs = []
        for p in ps:
            with Image.open(p) as im:
                xs.append(TF(im.convert('RGB')))
        return torch.stack(xs)

    print('[kmeans] 载入图...', flush=True)
    xf, xe = load(fit_p), load(ev_p)
    print('[kmeans] fit %s  eval %s' % (tuple(xf.shape), tuple(xe.shape)), flush=True)

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model, _ = vqvae_from_ckpt(ck)
    model.eval()
    print('[kmeans] %s step=%s n_codes=%s' % (
        os.path.basename(a.ckpt), ck.get('step'), ck.get('n_codes')), flush=True)

    @torch.no_grad()
    def enc_z(xs):
        out = []
        for s in range(0, len(xs), 8):
            out.append(model.enc(xs[s:s + 8]))
        return torch.cat(out, 0)

    @torch.no_grad()
    def dec_z(z):
        out = []
        for s in range(0, len(z), 8):
            out.append(model.dec(z[s:s + 8]).clamp(-1, 1))
        return torch.cat(out, 0)

    t0 = time.time()
    zf = enc_z(xf)                                    # (Nf, C, H, W)
    ze = enc_z(xe)
    Cf, H, W = zf.shape[1], zf.shape[2], zf.shape[3]
    Zf = zf.permute(0, 2, 3, 1).reshape(-1, Cf)       # (Nf*H*W, C)
    Ze = ze.permute(0, 2, 3, 1).reshape(-1, Cf)
    print('[kmeans] 编码完成 %.1fs  latent %s -> %s' % (time.time() - t0, tuple(zf.shape), tuple(Zf.shape)), flush=True)

    # 参考基线：原码本（1024，P 自己训过的）
    print()
    print('%-14s %10s %10s %10s' % ('codebook', 'PSNR', 'quant-err', 'notes'))
    print('-' * 52)

    @torch.no_grad()
    def eval_with_codebook(W):
        """把码本换成 W，然后编码->查表->解码，在留出图上算 PSNR。

        ⚠️ 不能用 codebook.weight.data.copy_(W)：K 不同时形状不匹配会报
        "The size of tensor a (1024) must match the size of tensor b (2048)"。
        必须整体换掉 nn.Embedding 模块。
        """
        import torch.nn
        old_mod = model.vq.codebook
        new_mod = nn.Embedding(W.shape[0], W.shape[1])
        new_mod.weight.data.copy_(W)
        new_mod.weight.requires_grad_(False)
        model.vq.codebook = new_mod
        try:
            q = model.vq.quantize_only(ze)             # (B, C, H, W)
            rec = dec_z(q)
            err = float(((ze - q) ** 2).mean())
        finally:
            model.vq.codebook = old_mod                # 还原
        return psnr(xe, rec), err

    base_W = model.vq.codebook.weight.data.clone()
    P0, E0 = eval_with_codebook(base_W)
    print('%-14s %10.2f %10.4f %10s' % ('P trained(%d)' % base_W.shape[0], P0, E0, '参考'))

    for K in [int(k) for k in a.ks.split(',')]:
        t1 = time.time()
        W, n_empty = kmeans(Zf, K)
        P1, E1 = eval_with_codebook(W)
        print('%-14s %10.2f %10.4f %10s' % (
            'kmeans K=%d' % K, P1, E1,
            '%+.2f dB  %.0fs' % (P1 - P0, time.time() - t1)))
        print('%-14s %10s %10s %10s' % ('', '', '', 'empty=%d' % n_empty))

    print()
    print('⇒ 若 K 增大而 PSNR 基本不动 ⇒ 限制【不是】值量化粒度，加码本没用。')
    print('   若随 K 明显上升 ⇒ 加码本有效（且真重训会更好，因为上面是下界）。')


if __name__ == '__main__':
    main()
