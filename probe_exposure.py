# -*- coding: utf-8 -*-
"""probe_exposure.py — 裁决"AR 在 T=1024 上到底缺什么"。

| 模式 | 每一步的输入前缀 | 含义 |
|---|---|---|
| teacher-forced | **真值 token** | 光看模型的【每步分布】好不好 |
| free-running    | 模型自己采出来的 | 加上【误差累积】之后的效果 |

因为模型是因果的，teacher-forced 只需 **一次前向**：
forward_logits(x_true)[:, i] 就是"给定前 i 个真 token"时的分布。

两者的 top-k / temperature / 解码器完全相同 ⇒ 唯一的变量是"前缀是真是假"。
若 teacher-forced 明显更锐 ⇒ 病根是误差累积（exposure bias）；
若两者一样灰 ⇒ 病根在每步分布本身（欠训/容量）。
"""
import argparse, math, os, sys, time
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_vqvae import vqvae_from_ckpt
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2


def load_ar(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, cfg


def topk_sample(lg, k, temp):
    lg = lg / max(temp, 1e-6)
    v, _ = torch.topk(lg, k, dim=-1)
    thresh = v[..., -1, None]
    lg = torch.where(lg < thresh, torch.full_like(lg, float('-inf')), lg)
    probs = F.softmax(lg, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, 1).reshape(probs.shape[:-1])


def decode_grid(vqvae, toks, side, out):
    vq, _ = vqvae_from_ckpt(vqvae)
    idx = toks[:, :side * side].view(-1, side, side)
    with torch.no_grad():
        emb = vq.vq.codebook(idx)
        z = emb.permute(0, 3, 1, 2).contiguous()
        img = vq.dec(z)
    n = img.shape[0]
    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * 256, rows * 256), (20, 20, 20))
    for i in range(n):
        r, c = divmod(i, cols)
        t = ((img[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
        canvas.paste(Image.fromarray(t), (c * 256, r * 256))
    canvas.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', required=True)
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--tokens', required=True, help='(N, T) 真值 token 文件')
    ap.add_argument('--start', type=int, default=27658, help='验证区间起点')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-prefix', required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model, cfg = load_ar(a.ar)
    T = cfg.total_tokens
    side = int(round(T ** 0.5))
    print('[probe] cfg: T=%d grid=%d cond=%s' % (T, side, getattr(cfg, 'cond_mode', '?')))

    alltok = torch.load(a.tokens, map_location='cpu', weights_only=False).long()
    true = alltok[a.start:a.start + a.n]
    print('[probe] true tokens %s from rows [%d,%d)' % (tuple(true.shape), a.start, a.start + a.n))

    # ---- (A) teacher-forced：一次前向即可（因果）----
    t0 = time.perf_counter()
    with torch.no_grad():
        # forward_logits 返回 (think_logits, gen_logits)；
        # head-mode=flat 时 gen_logits 直接是 (B,T,vocab)；factorized 时是 (cls, code) 二元组。
        _think, gen = model.forward_logits(true, None)
        if isinstance(gen, (tuple, list)):
            cl, cd = gen
            cmap = (None, None)
            lg = model.merge(cl, cd, cmap) if hasattr(model, 'merge') else cl
        else:
            lg = gen
        assert lg.dim() == 3 and lg.shape[:2] == true.shape, 'logits 形状异常: %s' % (tuple(lg.shape),)
        tf_toks = topk_sample(lg, a.top_k, a.temperature)
    print('[probe] teacher-forced sampled in %.1fs' % (time.perf_counter() - t0))
    agree = float((tf_toks == true).float().mean())
    print('[probe] teacher-forced 与真值逐位一致率: %.1f%%  (1/一致率 ≈ 每步平均候选数)'
          % (agree * 100))
    print('[probe]   注: CE=%.2f nats ⇒ 理论一致率上界 %.1f%%'
          % (float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), true.reshape(-1))),
             math.exp(-float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), true.reshape(-1)))) * 100))

    # ---- (B) free-running：走训练时一致的那条采样路径 ----
    t0 = time.perf_counter()
    with torch.no_grad():
        fr_toks = model.generate(None, steps=T, top_k=a.top_k,
                                 temperature=a.temperature, seed=a.seed, batch=a.n)
    print('[probe] free-running sampled in %.1fs' % (time.perf_counter() - t0))

    for name, tk in (('TF', tf_toks), ('FREE', fr_toks)):
        flat = tk.reshape(-1)
        used = int(torch.unique(flat).numel())
        cnt = torch.bincount(flat.long(), minlength=cfg.vocab_size).float()
        p = cnt / cnt.sum()
        ent = float(-(p[p > 0] * p[p > 0].log()).sum())
        print('[probe] %-5s 码本 %d/%d (%.1f%%)  熵 %.3f/%.3f (%.1f%%)'
              % (name, used, cfg.vocab_size, used / cfg.vocab_size * 100,
                 ent, math.log(cfg.vocab_size), ent / math.log(cfg.vocab_size) * 100))

    decode_grid(a.vqvae, true, side, a.out_prefix + '_TRUE.png')
    decode_grid(a.vqvae, tf_toks, side, a.out_prefix + '_TF.png')
    decode_grid(a.vqvae, fr_toks, side, a.out_prefix + '_FREE.png')
    print('[probe] wrote %s_{TRUE,TF,FREE}.png' % a.out_prefix)


if __name__ == '__main__':
    main()
