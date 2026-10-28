# -*- coding: utf-8 -*-
"""train_ar.py - 用 VQ-VAE token 预训练小 AR 模型，记录速度/显存/Loss。"""
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse, json, os, sys, time, threading
import torch
import torch.nn.functional as F
import psutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel

def rss_gb(): return psutil.Process().memory_info().rss/1e9
class Peak:
    def __init__(self): self.p=rss_gb(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set(): self.p=max(self.p,rss_gb()); time.sleep(0.02)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p

def build_model(a, cfg):
    if a.attn == 'kda':
        from scripts.lib.kda_model import SmallARImageModelKDA
        cfg.chunk_size = a.chunk_size
        return SmallARImageModelKDA(cfg)
    return SmallARImageModel(cfg)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--attn', default='mha', choices=['mha','kda'])
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--n_layer', type=int, default=10)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--ffn_dim', type=int, default=2048)
    ap.add_argument('--total_tokens', type=int, default=256)
    ap.add_argument('--chunk_size', type=int, default=128)
    ap.add_argument('--think_weight', type=float, default=0.1)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--ckpt_every', type=int, default=500)
    a=ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    tokens=torch.load(a.tokens, map_location='cpu')
    if tokens.ndim != 2:
        tokens=tokens.reshape(tokens.shape[0], -1)
    N = tokens.shape[0]
    cfg=SmallImageConfig(d_model=a.d_model,n_layer=a.n_layer,n_head=a.n_head,
                         ffn_dim=a.ffn_dim,total_tokens=a.total_tokens)
    model=build_model(a,cfg)
    nparam=sum(p.numel() for p in model.parameters())
    print(f'[ar] attn={a.attn} tokens={tuple(tokens.shape)} params={nparam/1e6:.2f}M batch={a.batch}',flush=True)
    opt=torch.optim.AdamW(model.parameters(), lr=a.lr) if a.attn=='mha' else __import__('bitsandbytes').optim.AdamW8bit(model.parameters(), lr=a.lr)
    text=torch.zeros(a.batch, cfg.text_dim)
    peak=Peak(); peak.start()
    losses=[]; times=[]; t0=time.time()
    for step in range(1,a.steps+1):
        idx=torch.randint(0,N,(a.batch,))
        x=tokens[idx].long()
        ts=time.perf_counter()
        think_logits, gen_logits = model(x, text)
        gen_loss=F.cross_entropy(gen_logits.reshape(-1,cfg.vocab_size), x.reshape(-1))
        think_loss=F.cross_entropy(think_logits.reshape(-1,cfg.vocab_size), x[:,:cfg.think_tokens].reshape(-1))
        loss=gen_loss + a.think_weight*think_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        dt=time.perf_counter()-ts; losses.append(float(loss.detach())); times.append(dt)
        if step%20==0 or step==1:
            rec=dict(step=step, loss=losses[-1], gen=float(gen_loss.detach()), think=float(think_loss.detach()),
                     step_s=dt, elapsed_s=time.time()-t0, rss_gb=rss_gb())
            with open(os.path.join(a.out,'train.jsonl'),'a',encoding='utf-8') as f:
                f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            print(f'[ar step {step}] loss={rec["loss"]:.4f} gen={rec["gen"]:.4f} think={rec["think"]:.4f} {dt:.3f}s rss={rec["rss_gb"]:.2f}GB',flush=True)
        if step%a.ckpt_every==0:
            torch.save(dict(model=model.state_dict(), step=step, cfg=cfg.__dict__), os.path.join(a.out,f'ckpt_{step}.pt'))
            torch.save(dict(model=model.state_dict(), step=step, cfg=cfg.__dict__), os.path.join(a.out,'ckpt_last.pt'))
    peak_rss=peak.done()
    summary=dict(kind='ar_pretrain', attn=a.attn, params=int(nparam), steps=a.steps,
                 mean_step_s=sum(times)/len(times), first_loss=losses[0], last_loss=losses[-1],
                 mean_last20=sum(losses[-20:])/min(20,len(losses)), peak_rss_gb=float(peak_rss),
                 tokens_file=a.tokens)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    with open(os.path.join(a.out,'summary.json'),'w',encoding='utf-8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)

if __name__=='__main__': main()