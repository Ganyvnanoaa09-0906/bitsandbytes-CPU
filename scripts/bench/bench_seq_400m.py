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
import sys, time, json, torch, torch.nn.functional as F
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
out=[]
for T in [16,64,128,256]:
    cfg=SmallImageConfig(d_model=1280,n_layer=16,n_head=16,ffn_dim=7168,total_tokens=256)
    m=SmallARImageModel(cfg); opt=bnb.optim.AdamW8bit(m.parameters(),lr=1e-4)
    x=torch.randint(0,1024,(1,T)); text=torch.zeros(1,768); ts=[]
    for _ in range(2):
        t0=time.perf_counter(); th,ge=m(x,text); loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th[:,:min(16,T)].reshape(-1,1024),x[:,:min(16,T)].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ts.append(time.perf_counter()-t0)
    rec=dict(T=T,step_times=ts,mean=sum(ts)/len(ts)); out.append(rec); print(json.dumps(rec,ensure_ascii=False))