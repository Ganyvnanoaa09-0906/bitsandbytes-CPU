import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
# -*- coding: utf-8 -*-
import json, os, sys, time, threading, gc
import torch, torch.nn.functional as F, psutil
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb

def rss(): return psutil.Process().memory_info().rss/1e9
class Peak:
    def __init__(self): self.p=rss(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set(): self.p=max(self.p,rss()); time.sleep(0.02)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p

CFGS=[
 dict(name='400M_wide', d_model=1536,n_layer=14,n_head=16,ffn_dim=6144),
 dict(name='400M_deep', d_model=1280,n_layer=16,n_head=16,ffn_dim=7168),
 dict(name='410M_deep', d_model=1408,n_layer=16,n_head=16,ffn_dim=6144),
]
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
tokens=torch.randint(0,1024,(1,256)); text=torch.zeros(1,768)
out=[]
for c in CFGS:
    name=c.pop('name'); cfg=SmallImageConfig(**c,total_tokens=256); model=SmallARImageModel(cfg)
    n=sum(p.numel() for p in model.parameters()); opt=bnb.optim.AdamW8bit(model.parameters(),lr=1e-4)
    peak=Peak(); peak.start(); ts=[]
    try:
        for _ in range(6):
            t0=time.perf_counter(); th,ge=model(tokens,text)
            loss=F.cross_entropy(ge.reshape(-1,1024),tokens.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),tokens[:,:16].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ts.append(time.perf_counter()-t0)
    except RuntimeError as e:
        out.append(dict(name=name,params=int(n),error=str(e)[:200])); print(out[-1],flush=True); del model,opt; gc.collect(); continue
    pk=peak.done(); rec=dict(name=name,params=int(n),step_times=ts,mean_last4=sum(ts[-4:])/4,peak_rss_gb=float(pk))
    out.append(rec); print(json.dumps(rec,ensure_ascii=False),flush=True); del model,opt; gc.collect()
with open(r'D:\work\bitsandbytes-CPU\.autopilot\dense_400m_bench.json','w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
print('===DONE===')