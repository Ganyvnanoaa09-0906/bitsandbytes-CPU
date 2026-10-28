# -*- coding: utf-8 -*-
"""kda_model.py - KDA 风格 chunked gated linear attention + 循环小生图模型原型。

用 chunkwise GLA（gated linear attention）近似 KDA/Delta 类线性注意力：
- 每个 chunk 内做因果 gated attention，chunk 间传递 state；
- O(T * C * D) 级别，CPU/AVX-2 友好；
- 可与 full causal MHA 对比 loss/速度。
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
import math
import os, sys
import torch
import torch.nn
import torch.nn.functional
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes"))
import bitsandbytes

class KDAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, chunk_size: int = 64, use_qk_norm: bool = True):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.chunk_size = chunk_size
        self.use_qk_norm = use_qk_norm
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.g_proj = nn.Linear(d_model, n_head, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _chunk(self, q, k, v, g):
        # q,k,v: [B,H,T,Dh]; g: [B,H,T,1]
        B, H, T, Dh = q.shape
        C = self.chunk_size
        G = g.cumsum(dim=2)  # [B,H,T,1]
        out = torch.zeros_like(v)
        state = q.new_zeros(B, H, Dh, Dh)
        for start in range(0, T, C):
            end = min(start + C, T)
            qc = q[:, :, start:end]
            kc = k[:, :, start:end]
            vc = v[:, :, start:end]
            Gc = G[:, :, start:end]              # [B,H,L,1]
            Gs = G[:, :, start:start + 1]        # [B,H,1,1]
            L = end - start
            # intra-chunk: i >= j
            attn = torch.einsum('bhid,bhjd->bhij', qc, kc)  # [B,H,L,L]
            idx = torch.arange(L, device=q.device)
            causal = (idx[:, None] >= idx[None, :]).float()
            decay = torch.exp(torch.clamp(Gc - Gc.transpose(-2, -1), max=0.0))  # [B,H,L,L], avoid overflow on upper triangle
            attn = attn * causal * decay
            oc = torch.einsum('bhij,bhjd->bhid', attn, vc)
            # inter-chunk state
            qs = qc * torch.exp(Gc - Gs)
            oc = oc + torch.einsum('bhid,bhde->bhie', qs, state)
            # state update to chunk end
            Ge = G[:, :, end - 1:end]
            ks = kc * torch.exp(Ge - Gc)
            state = state * torch.exp(Ge - Gs) + torch.einsum('bhjd,bhje->bhde', ks, vc)
            out[:, :, start:end] = oc
        return out

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        g = -F.softplus(self.g_proj(x))  # [B,T,H]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        g = g.transpose(1, 2).unsqueeze(-1)  # [B,H,T,1]
        if self.use_qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        y = self._chunk(q, k, v, g)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)
class KDABlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_dim, chunk_size=64, use_qk_norm=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = KDAttention(d_model, n_head, chunk_size, use_qk_norm)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SmallARImageModelKDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.total_tokens, cfg.d_model)
        self.cond_proj = nn.Linear(cfg.text_dim, cfg.d_model)
        n_head = getattr(cfg, "n_head", 8)
        chunk_size = getattr(cfg, "chunk_size", 64)
        use_qk_norm = getattr(cfg, "use_qk_norm", True)
        self.blocks = nn.ModuleList([KDABlock(cfg.d_model, n_head, cfg.ffn_dim, chunk_size, use_qk_norm)
                                     for _ in range(cfg.n_layer)])
        self.think_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.gen_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.loop_idx = [i for i in range(cfg.loop_start, min(cfg.loop_end, cfg.n_layer - 1) + 1)]
        self.inject_gate = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, text_cond, return_hidden=False):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x0 = self.token_emb(tokens) + self.pos_emb(pos) + self.cond_proj(text_cond).unsqueeze(1)
        x = x0
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.loop_idx:
                for _ in range(max(1, self.cfg.loop_times) - 1):
                    x = blk(x)
            x = x + self.inject_gate * x0  # input injection
        if return_hidden:
            return x
        return self.think_head(x[:, :self.cfg.think_tokens]), self.gen_head(x)

    @torch.no_grad()
    def generate(self, text_cond, steps=None, top_k=50, temperature=1.0,
                 guidance_scale=1.0, uncond_text=None):
        cfg = self.cfg
        B = text_cond.shape[0]
        use_cfg = guidance_scale is not None and guidance_scale > 1.0
        if use_cfg:
            if uncond_text is None:
                uncond_text = torch.zeros_like(text_cond)
        x = torch.zeros(B, 1, dtype=torch.long, device=text_cond.device)
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
def _main():
    import argparse, json, time, threading, psutil, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from small_image_model import SmallImageConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=10)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--ffn_dim", type=int, default=2048)
    ap.add_argument("--total_tokens", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    torch.set_num_threads(6)
    cfg = SmallImageConfig(d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head,
                           ffn_dim=a.ffn_dim, total_tokens=a.total_tokens)
    cfg.chunk_size = a.chunk_size; cfg.use_qk_norm = True
    model = SmallARImageModelKDA(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[kda] params={n_param/1e6:.2f}M chunk={a.chunk_size}", flush=True)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr)
    tokens = torch.randint(0, cfg.vocab_size, (a.batch, cfg.total_tokens))
    text = torch.randn(a.batch, cfg.text_dim)
    # quick generation test (short)
    gen = model.generate(text, steps=4, guidance_scale=1.0)
    print("[kda] gen smoke", tuple(gen.shape), flush=True)
    losses, times = [], []
    for step in range(1, a.steps + 1):
        t0 = time.perf_counter()
        think_logits, gen_logits = model(tokens, text)
        loss = F.cross_entropy(think_logits.reshape(-1, cfg.vocab_size), tokens[:, :cfg.think_tokens].reshape(-1))
        loss = loss + F.cross_entropy(gen_logits.reshape(-1, cfg.vocab_size), tokens.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach())); times.append(time.perf_counter() - t0)
        print(f"[kda step {step}] loss={losses[-1]:.4f} time={times[-1]:.2f}s rss={psutil.Process().memory_info().rss/1e9:.2f}GB", flush=True)
    print(json.dumps(dict(kind="kda_smoke", params=int(n_param), chunk_size=a.chunk_size,
                          steps=a.steps, losses=losses, times_s=times, mean_step_s=sum(times)/len(times)),
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _main()
