import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
# -*- coding: utf-8 -*-
import json, os, sys, time, threading, torch, torch.nn.functional, psutil, gc
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes
def rss(): return psutil.Process().memory_info().rss/1e9
class Peak:
    def __init__(self): self.p=rss(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set(): self.p=max(self.p,rss()); time.sleep(0.02)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p
CFGS=[
 dict(name='1792x10', d_model=1792,n_layer=10,n_head=16,ffn_dim=7168),
 dict(name='2048x8', d_model=2048,n_layer=8,n_head=16,ffn_dim=8192),
 dict(name='1024x22', d_model=1024,n_layer=22,n_head=16,ffn_dim=6144),
 dict(name='1536x12', d_model=1536,n_layer=12,n_head=16,ffn_dim=8192),
 dict(name='1280x16', d_model=1280,n_layer=16,n_head=16,ffn_dim=7168),
]
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
x=torch.randint(0,1024,(1,256)); text=torch.zeros(1,768)
out=[]
for c in CFGS:
    name=c.pop('name'); cfg=SmallImageConfig(**c,total_tokens=256); m=SmallARImageModel(cfg)
    n=sum(p.numel() for p in m.parameters()); opt=bnb.optim.AdamW8bit(m.parameters(),lr=1e-4)
    peak=Peak(); peak.start(); ts=[]
    try:
        for _ in range(4):
            t0=time.perf_counter(); th,ge=m(x,text); loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ts.append(time.perf_counter()-t0)
        rec=dict(name=name,params=int(n),step_times=ts,mean_last3=sum(ts[-3:])/3,peak_rss_gb=float(peak.done()))
    except Exception as e:
        rec=dict(name=name,params=int(n),error=f'{type(e).__name__}: {str(e)[:200]}')
    out.append(rec); print(json.dumps(rec,ensure_ascii=False),flush=True); del m,opt; gc.collect()
with open(r'D:\work\bitsandbytes-CPU\.autopilot\shape_bench_400m.json','w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)