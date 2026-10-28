# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys, time, argparse, torch, torch.nn.functional as F
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb
ap=argparse.ArgumentParser(); ap.add_argument('--size',default='400M'); ap.add_argument('--steps',type=int,default=3); a=ap.parse_args()
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
if a.size=='33M': cfg=SmallImageConfig(d_model=512,n_layer=10,n_head=8,ffn_dim=2048,total_tokens=256)
else: cfg=SmallImageConfig(d_model=1280,n_layer=16,n_head=16,ffn_dim=7168,total_tokens=256)
model=SmallARImageModel(cfg); opt=bnb.optim.AdamW8bit(model.parameters(),lr=1e-4)
x=torch.randint(0,1024,(1,256)); text=torch.zeros(1,768)
def step():
    th,ge=model(x,text)
    loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
for _ in range(2): step()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
    for _ in range(a.steps): step()
print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=35))