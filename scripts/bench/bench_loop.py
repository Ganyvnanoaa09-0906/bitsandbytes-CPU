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
import sys, time, json, torch, torch.nn.functional as F
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
for size,cfgbase in [('33M',dict(d_model=512,n_layer=10,n_head=8,ffn_dim=2048)),('400M',dict(d_model=1280,n_layer=16,n_head=16,ffn_dim=7168))]:
    for lt in [1,2]:
        cfg=SmallImageConfig(**cfgbase,total_tokens=256,loop_times=lt)
        m=SmallARImageModel(cfg); opt=bnb.optim.AdamW8bit(m.parameters(),lr=1e-4)
        x=torch.randint(0,1024,(1,256)); text=torch.zeros(1,768); ts=[]
        for _ in range(2):
            t0=time.perf_counter(); th,ge=m(x,text); loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ts.append(time.perf_counter()-t0)
        print(json.dumps(dict(size=size,loop_times=lt,params=sum(p.numel() for p in m.parameters()),step_times=ts,mean=sum(ts)/len(ts)),ensure_ascii=False))