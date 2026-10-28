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
import sys, time, json, argparse, torch, torch.nn.functional as F, psutil, threading
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb
ap=argparse.ArgumentParser(); ap.add_argument('--cls',default='Adam8bit'); ap.add_argument('--bits',type=int,default=8); ap.add_argument('--steps',type=int,default=2); a=ap.parse_args()
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
cfg=SmallImageConfig(d_model=1280,n_layer=16,n_head=16,ffn_dim=7168,total_tokens=256)
m=SmallARImageModel(cfg); n=sum(p.numel() for p in m.parameters())
cls=getattr(bnb.optim,a.cls)
try:
    opt=cls(m.parameters(),lr=1e-4, optim_bits=a.bits)
except TypeError:
    opt=cls(m.parameters(),lr=1e-4)
x=torch.randint(0,1024,(1,256)); text=torch.zeros(1,768)
def rss(): return psutil.Process().memory_info().rss/1e9
class Peak:
    def __init__(self): self.p=rss(); self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set(): self.p=max(self.p,rss()); time.sleep(0.02)
    def start(self): self.t.start()
    def done(self): self.stop.set(); self.t.join(0.2); return self.p
peak=Peak(); peak.start(); ts=[]
for _ in range(a.steps):
    t0=time.perf_counter(); th,ge=m(x,text); loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ts.append(time.perf_counter()-t0)
print(json.dumps(dict(cls=a.cls,bits=a.bits,params=int(n),step_times=ts,mean=sum(ts)/len(ts),peak_rss_gb=float(peak.done())),ensure_ascii=False))