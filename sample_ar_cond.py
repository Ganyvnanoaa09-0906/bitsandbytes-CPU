# -*- coding: utf-8 -*-
"""sample_ar_cond.py — 决定性诊断：让 AR 模型"看着真实的前缀"续画。

区分两种失败：
  能续出结构  → 模型学到了条件分布，只是 exposure bias（自回归误差累积）
                → 解法：调度采样 / 更多训练 / 更稳的位置编码
  续不出结构  → 模型根本没学到空间结构
                → 解法：加容量 / 加数据 / 改目标

做法：从 tokens_final2.pt 取真实图，保留前 50% token，
      让模型续画后 50%，再经 VQ-VAE 解码。
      上排 = 原图重建（tokenizer 天花板），下排 = 前50%真实 + 后50%模型生成。
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2, run_blocks
from small_vqvae import VQVAE, vqvae_from_ckpt


def gen_with_prefix(model, tokens, frac, top_k, temperature, seed):
    """tokens: (B,T) 真实 token。保留前 frac 比例，续画其余。"""
    cfg = model.cfg
    B, T = tokens.shape
    n_pre = max(1, int(T * frac))
    prefix = tokens[:, :n_pre].clone()

    def forward_logits_last(tok, caches, pos):
        x = model.embed(tok, pos_offset=pos)
        x = run_blocks(model.blocks, x, model.loop_idx, cfg.loop_times, None, None, caches)
        cl, cd = model.gen_head.logits(x[:, -1:])
        return model.merge(cl, cd, None)[:, -1]

    caches = [dict(k=None, v=None) for _ in range(cfg.n_layer)]
    logits = forward_logits_last(prefix, caches, 0)

    out = prefix
    pos = n_pre
    for _ in range(T - n_pre):
        lg = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            v, _ = torch.topk(lg, min(top_k, lg.shape[-1]), dim=-1)
            lg = lg.masked_fill(lg < v[:, [-1]], -1e9)
        nxt = torch.multinomial(F.softmax(lg, dim=-1), 1)
        out = torch.cat([out, nxt], 1)
        logits = forward_logits_last(nxt, caches, pos)
        pos += 1
    return out, n_pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--tokens', required=True, help='tokens_final2.pt')
    ap.add_argument('--out', default='cond.png')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--frac', type=float, default=0.5, help='保留多少比例的真实前缀')
    ap.add_argument('--top-k', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)

    ck = torch.load(a.ar, map_location='cpu', weights_only=False)
    cfgd = ck.get('cfg')
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    model = SmallARImageModelV2(cfg)
    # loop Transformer: mount loop_mods BEFORE loading, order matters.
    # loop_cache=perloop is the semantics training used (see sample_ar.py help).
    sd = ck['model']
    if any(k.startswith('loop_mods.') for k in sd):
        from loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith('step_enc.table.weight') or k.endswith('cross_res.gate'):
                mx = v.shape[0]
                break
        lmax = max(1, mx - 1)
        model.set_loop(LoopConfig(loop_start=cfg.loop_start,
                                  loop_end=cfg.loop_end, loop_L=lmax,
                                  loop_L_max=lmax, loop_random=False,
                                  loop_cache='perloop'))
        print('  [loop] looped ckpt detected: L_max=%d, infer L=%d, cache=perloop'
              % (lmax, lmax))
    model.load_state_dict(sd)
    model.eval()
    print('AR: %.2fM  cfg grid=%d total_tokens=%d loop_times=%d'
          % (sum(p.numel() for p in model.parameters()) / 1e6,
             cfg.grid, cfg.total_tokens, cfg.loop_times))

    allt = torch.load(a.tokens, map_location='cpu')
    if allt.ndim != 2:
        allt = allt.reshape(allt.shape[0], -1)
    idx = torch.randperm(allt.shape[0])[:a.n]
    real = allt[idx]
    print('真实 token: %s   前缀比例 %.0f%%' % (tuple(real.shape), a.frac * 100))

    t0 = time.perf_counter()
    with torch.no_grad():
        cont, n_pre = gen_with_prefix(model, real, a.frac, a.top_k, a.temperature, a.seed)
    print('续画完成 %.1fs  前缀 %d token' % (time.perf_counter() - t0, n_pre))

    vk = torch.load(a.vqvae, map_location='cpu', weights_only=False)
    ch = tuple(vk['ch']) if vk.get('ch') else (64, 128, 256, 256)
    vq = VQVAE(embed_dim=64, n_codes=int(vk.get('n_codes', 1024)), ch=ch)
    vq.load_state_dict(vk['model'])
    vq.eval()
    side = cfg.grid

    def decode(toks):
        i = toks[:, :side * side].view(-1, side, side)
        with torch.no_grad():
            emb = vq.vq.codebook(i)
            z = emb.permute(0, 3, 1, 2).contiguous()
            return vq.dec(z)

    with torch.no_grad():
        img_real = decode(real)
        img_cont = decode(cont)

    from PIL import Image
    C = 256
    n = img_real.shape[0]
    canvas = Image.new('RGB', (n * C, 2 * C), (20, 20, 20))
    for i in range(n):
        t = ((img_real[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
        canvas.paste(Image.fromarray(t), (i * C, 0))
        t2 = ((img_cont[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
        canvas.paste(Image.fromarray(t2), (i * C, C))
    canvas.save(a.out)
    print('  上排 = 原图重建（天花板）')
    print('  下排 = 前 %.0f%% 真实 + 后 %.0f%% 模型生成' % (a.frac * 100, (1 - a.frac) * 100))
    print('-> %s' % a.out)

    # 前缀越短，越接近纯生成
    for f in (0.75, 0.5, 0.25, 0.0):
        with torch.no_grad():
            cc, np_ = gen_with_prefix(model, real[:3], f, a.top_k, a.temperature, a.seed)
        with torch.no_grad():
            im = decode(cc)
        cv = Image.new('RGB', (3 * C, C), (20, 20, 20))
        for i in range(3):
            t = ((im[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
            cv.paste(Image.fromarray(t), (i * C, 0))
        p = a.out.replace('.png', '_f%02d.png' % int(f * 100))
        cv.save(p)
        print('  frac=%.0f%% -> %s' % (f * 100, p))


if __name__ == '__main__':
    main()
