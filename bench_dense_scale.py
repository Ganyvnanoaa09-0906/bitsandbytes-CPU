# -*- coding: utf-8 -*-
"""bench_dense_scale.py - dense 模型缩放速度基准（R5 CPU, fp32, seq=256）。"""
from __future__ import annotations
import json, os, sys, time, threading, gc
import torch
import torch.nn.functional as F
import psutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes"))
import bitsandbytes as bnb

def rss_gb(): return psutil.Process().memory_info().rss/1e9
class Peak:
    def __init__(self): self.p=rss_gb(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set(): self.p=max(self.p,rss_gb()); time.sleep(0.02)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p

CONFIGS=[
    dict(name='33M', d_model=512, n_layer=10, n_head=8, ffn_dim=2048),
    dict(name='70M', d_model=640, n_layer=12, n_head=8, ffn_dim=2560),
    dict(name='130M', d_model=768, n_layer=14, n_head=12, ffn_dim=3072),
    dict(name='250M', d_model=1024, n_layer=14, n_head=16, ffn_dim=4096),
    dict(name='400M', d_model=1280, n_layer=14, n_head=16, ffn_dim=5120),
]

def main():
    torch.set_num_threads(6)
    torch.backends.mkldnn.enabled=True
    tokens=torch.randint(0,1024,(1,256))
    text=torch.zeros(1,768)
    out=[]
    for c in CONFIGS:
        name=c.pop('name')
        cfg=SmallImageConfig(d_model=c['d_model'],n_layer=c['n_layer'],n_head=c['n_head'],
                             ffn_dim=c['ffn_dim'],total_tokens=256)
        model=SmallARImageModel(cfg)
        nparam=sum(p.numel() for p in model.parameters())
        opt=bnb.optim.AdamW8bit(model.parameters(),lr=1e-4)
        peak=Peak(); peak.start()
        times=[]
        try:
            for step in range(7):
                t0=time.perf_counter()
                think,gen=model(tokens,text)
                loss=F.cross_entropy(gen.reshape(-1,1024),tokens.reshape(-1)) + 0.1*F.cross_entropy(think.reshape(-1,1024),tokens[:,:16].reshape(-1))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                times.append(time.perf_counter()-t0)
        except RuntimeError as e:
            out.append(dict(name=name,params=int(nparam),error=str(e)[:200])); print(out[-1],flush=True)
            del model,opt; gc.collect(); continue
        pk=peak.done()
        rec=dict(name=name,params=int(nparam),step_times=times,mean_last5=sum(times[-5:])/5,
                 peak_rss_gb=float(pk))
        out.append(rec); print(json.dumps(rec,ensure_ascii=False),flush=True)
        del model,opt; gc.collect()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'.autopilot','dense_scale_bench.json'),'w',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print('===DONE===',flush=True)

if __name__=='__main__': main()