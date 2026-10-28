import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
# -*- coding: utf-8 -*-
import argparse, json, sys, time, threading, torch, torch.nn.functional as F, psutil
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb
ap=argparse.ArgumentParser()
ap.add_argument('--batch', type=int, default=1)
ap.add_argument('--opt', default='adamw8bit', choices=['adamw8bit','adamw'])
ap.add_argument('--steps', type=int, default=3)
ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--optim_bits', type=int, default=32)
a=ap.parse_args()
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
torch.set_num_threads(a.threads); torch.backends.mkldnn.enabled=True
def rss(): return psutil.Process().memory_info().rss/1e9
class Peak:
    def __init__(self): self.p=rss(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set(): self.p=max(self.p,rss()); time.sleep(0.02)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p
cfg=SmallImageConfig(d_model=1280,n_layer=16,n_head=16,ffn_dim=7168,total_tokens=256)
model=SmallARImageModel(cfg); n=sum(p.numel() for p in model.parameters())
opt=bnb.optim.AdamW8bit(model.parameters(),lr=1e-4) if a.opt=='adamw8bit' else torch.optim.AdamW(model.parameters(),lr=1e-4)
if a.opt=='adamw8bit': opt=bnb.optim.AdamW8bit(model.parameters(),lr=1e-4, optim_bits=a.optim_bits)
else: opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
x=torch.randint(0,1024,(a.batch,256)); text=torch.zeros(a.batch,768)
peak=Peak(); peak.start(); ts=[]
try:
    for _ in range(a.steps):
        t0=time.perf_counter(); th,ge=model(x,text)
        loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ts.append(time.perf_counter()-t0)
    rec=dict(batch=a.batch,opt=a.opt,params=int(n),step_times=ts,mean=sum(ts)/len(ts),peak_rss_gb=float(peak.done()))
except Exception as e:
    rec=dict(batch=a.batch,opt=a.opt,params=int(n),error=f'{type(e).__name__}: {str(e)[:200]}')
print(json.dumps(rec,ensure_ascii=False))
