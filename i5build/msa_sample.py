"""mask_mode='msa' 的逐步全量重算采样器（不能用 generate()：压缩摘要要整组到齐）。

为什么要单独写：多尺度注意力每次前向都必须看到【完整的组】才能算出行摘要，
增量 KV cache 给不了。而 generate() 走的是 cache 路径，注意力层检测到这一点会直接报错
（宁可报错也不静默退化成普通因果注意力）。

代价：每步重算整个前缀。T=256、30M 参数下约 0.3~0.5 s/步 ⇒ 每张图 80~130 s，
比带 cache 的 0.65 s 慢两个数量级 —— 但这一轮要的是【画质】而不是速度。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402


def sample(model, cfg, n, top_k, temperature, seed, cluster_map, device='cpu'):
    torch.manual_seed(seed)
    total = cfg.total_tokens
    bos_flag = bool(getattr(cfg, 'use_bos', False))
    toks = torch.zeros(n, total, dtype=torch.long, device=device)
    bos = torch.zeros(n, 1, dtype=torch.long, device=device)
    with torch.no_grad():
        for p in range(total):
            # 逐步全量重算：msa 的压缩组必须完整，给不了增量 cache
            inp = torch.cat([bos, toks[:, :p]], dim=1) if bos_flag else \
                (toks[:, :p] if p > 0 else bos)
            cl, cd = model.forward_logits(inp)[1]      # gen 头：(cluster, code)
            logits = model.merge(cl[:, -1], cd[:, -1], cluster_map)
            if temperature != 1.0:
                logits = logits / temperature
            if top_k and top_k < logits.shape[-1]:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            toks[:, p] = torch.multinomial(probs, 1).squeeze(-1)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--out', default='msa_samples.png')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    ck = torch.load(a.ar, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    model = SmallARImageModelV2(cfg)
    if any(k.startswith('loop_mods.') for k in ck['model']):
        from loop_transformer import LoopConfig
        mx = 4
        for k, v in ck['model'].items():
            if k.endswith('step_enc.table.weight') or k.endswith('cross_res.gate'):
                mx = v.shape[0]
                break
        model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                                  loop_L=max(1, mx - 1), loop_L_max=max(1, mx - 1),
                                  loop_random=False))
    model.load_state_dict(ck['model'])
    model.eval()
    print(f'AR: mask_mode={getattr(cfg, "mask_mode", "causal")} '
          f'msa_groups={getattr(cfg, "msa_groups", "")!r} '
          f'msa_local={getattr(cfg, "msa_local", 0)} '
          f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    t0 = time.perf_counter()
    cluster_map = (ck.get('cluster_of'), ck.get('offset_of'))
    toks = sample(model, cfg, a.n, a.top_k, a.temperature, a.seed, cluster_map)
    dt = time.perf_counter() - t0
    print(f'采样 {a.n} 张 x {cfg.total_tokens} 步：{dt:.1f}s ({dt/a.n:.1f}s/张)', flush=True)

    # 解码：走项目里已有的 vqvae_from_ckpt（从 checkpoint 读全部架构字段），
    # 不要自己猜类名/字段 —— 8x8 的 tokenizer 是 5 层 ch，写死 4 层会 state_dict 对不上。
    from vqvae_from_ckpt import vqvae_from_ckpt
    vq, _vk = vqvae_from_ckpt(a.vqvae)
    side = int(round(cfg.total_tokens ** 0.5))
    idx = toks[:, :side * side].view(-1, side, side)
    with torch.no_grad():
        emb = vq.vq.codebook(idx)                 # (B,side,side,C)
        z = emb.permute(0, 3, 1, 2).contiguous()  # (B,C,side,side)
        img = vq.dec(z)
    from PIL import Image
    n = img.shape[0]
    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * 256, rows * 256), (20, 20, 20))
    for i in range(n):
        r, c = divmod(i, cols)
        t = ((img[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
        canvas.paste(Image.fromarray(t), (c * 256, r * 256))
    canvas.save(a.out)
    print(f'-> {a.out}  ({canvas.width}x{canvas.height}, {n} 张)', flush=True)
    torch.save(toks, os.path.splitext(a.out)[0] + '.tokens.pt')


if __name__ == '__main__':
    sys.exit(main())
