# -*- coding: utf-8 -*-
"""small_image_model.py - R5 弹性小生图模型骨架（因果 Transformer + Think/Gen 双头）。

不绑定具体 VQ-VAE；可先做随机 token 训练冒烟，后续替换成真实 VQ-VAE token。
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
import argparse, json, os, sys, time, threading
from dataclasses import dataclass
import torch
import torch.nn
import torch.nn.functional
import psutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes"))
import bitsandbytes

@dataclass
class SmallImageConfig:
    vocab_size: int = 1024
    d_model: int = 512
    n_layer: int = 10
    n_head: int = 8
    ffn_dim: int = 2048
    total_tokens: int = 256
    think_tokens: int = 16
    text_dim: int = 768
    loop_start: int = 4
    loop_end: int = 7
    loop_times: int = 2

class CausalBlock(nn.Module):
    def __init__(self, cfg: SmallImageConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.ffn_dim), nn.GELU(),
                                 nn.Linear(cfg.ffn_dim, cfg.d_model))
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x
class SmallARImageModel(nn.Module):
    def __init__(self, cfg: SmallImageConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.total_tokens, cfg.d_model)
        self.cond_proj = nn.Linear(cfg.text_dim, cfg.d_model)
        self.blocks = nn.ModuleList([CausalBlock(cfg) for _ in range(cfg.n_layer)])
        self.think_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.gen_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.loop_idx = [i for i in range(cfg.loop_start, min(cfg.loop_end, cfg.n_layer - 1) + 1)]

    def forward(self, tokens, text_cond):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        x = x + self.cond_proj(text_cond).unsqueeze(1)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.loop_idx:
                for _ in range(max(1, self.cfg.loop_times) - 1):
                    x = blk(x)
        return self.think_head(x[:, :self.cfg.think_tokens]), self.gen_head(x)

    @torch.no_grad()
    @torch.no_grad()
    def generate(self, text_cond, steps=None, top_k=50, temperature=1.0,
                 guidance_scale=1.0, uncond_text=None):
        cfg = self.cfg
        use_cfg = guidance_scale is not None and guidance_scale > 1.0
        if use_cfg and uncond_text is None:
            uncond_text = torch.zeros_like(text_cond)
        x = torch.zeros(text_cond.shape[0], 1, dtype=torch.long, device=text_cond.device)
        for _ in range((steps or cfg.total_tokens) - 1):
            _, cond_logits = self.forward(x, text_cond)
            if use_cfg:
                _, uncond_logits = self.forward(x, uncond_text)
                logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            else:
                logits = cond_logits
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -1e9
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        return x
def _rss_gb():
    return psutil.Process().memory_info().rss / 1e9

class _Peak:
    def __init__(self):
        self.p = _rss_gb(); self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self.stop.is_set():
            self.p = max(self.p, _rss_gb()); time.sleep(0.01)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=10)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--ffn_dim", type=int, default=2048)
    ap.add_argument("--total_tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    torch.set_num_threads(6)
    cfg = SmallImageConfig(d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head,
                           ffn_dim=a.ffn_dim, total_tokens=a.total_tokens)
    model = SmallARImageModel(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[small] params={n_param/1e6:.2f}M config={cfg}", flush=True)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr)
    tokens = torch.randint(0, cfg.vocab_size, (a.batch, cfg.total_tokens))
    text = torch.randn(a.batch, cfg.text_dim)
    peak = _Peak(); peak.start()
    losses, times = [], []
    for step in range(1, a.steps + 1):
        t0 = time.perf_counter()
        think_logits, gen_logits = model(tokens, text)
        loss = F.cross_entropy(think_logits.reshape(-1, cfg.vocab_size), tokens[:, :cfg.think_tokens].reshape(-1))
        loss = loss + F.cross_entropy(gen_logits.reshape(-1, cfg.vocab_size), tokens.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach())); times.append(time.perf_counter() - t0)
        print(f"[small step {step}] loss={losses[-1]:.4f} time={times[-1]:.2f}s rss={_rss_gb():.2f}GB", flush=True)
    peak_rss = peak.done()
    print(json.dumps(dict(kind="small_ar_smoke", params=int(n_param), config=cfg.__dict__,
                          steps=a.steps, losses=losses, times_s=times,
                          mean_step_s=sum(times)/len(times), peak_rss_gb=float(peak_rss)),
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
