# -*- coding: utf-8 -*-
"""ab_loop_depth.py - 对比 loop=1 / loop=2 / 真实14层dense 的速度与 loss。"""
from __future__ import annotations
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
import argparse, json, os, sys, time, threading
import torch, torch.nn.functional as F, psutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes"))
import bitsandbytes as bnb
def rss(): return psutil.Process().memory_info().rss/1e9
class Peak:
    def __init__(self): self.p=rss(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set(): self.p=max(self.p,rss()); time.sleep(0.02)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p
ap=argparse.ArgumentParser(); ap.add_argument('--tokens',required=True); ap.add_argument('--steps',type=int,default=20); ap.add_argument('--batch',type=int,default=1); ap.add_argument('--threads',type=int,default=6); a=ap.parse_args()
torch.set_num_threads(a.threads); torch.backends.mkldnn.enabled=True
tokens=torch.load(a.tokens,map_location='cpu'); N=tokens.shape[0]
variants=[('loop1',10,1),('loop2',10,2),('dense14',14,1)]
out=[]
for name,n_layer,loop_times in variants:
    cfg=SmallImageConfig(d_model=512,n_layer=n_layer,n_head=8,ffn_dim=2048,total_tokens=256,loop_times=loop_times)
    if name=='dense14': cfg.loop_start=0; cfg.loop_end=-1
    m=SmallARImageModel(cfg); n=sum(p.numel() for p in m.parameters()); opt=bnb.optim.AdamW8bit(m.parameters(),lr=1e-4)
    x=torch.randint(0,1024,(a.batch,256)); text=torch.zeros(a.batch,768); ts=[]; losses=[]
    peak=Peak(); peak.start()
    for _ in range(a.steps):
        t0=time.perf_counter(); th,ge=m(x,text); loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ts.append(time.perf_counter()-t0); losses.append(float(loss.detach()))
    rec=dict(name=name,n_layer=n_layer,loop_times=loop_times,params=int(n),mean_step_s=sum(ts)/len(ts),first_loss=losses[0],last_loss=losses[-1],peak_rss_gb=float(peak.done()))
    out.append(rec); print(json.dumps(rec,ensure_ascii=False),flush=True); del m,opt
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'.autopilot','ab_loop_depth.json'),'w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)