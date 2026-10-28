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
# -*- coding: utf-8 -*-
import sys, time, json, argparse, torch, torch.nn, torch.nn.functional
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x): return self.weight * x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
def swap_norm(m):
    for name,ch in m.named_children():
        if isinstance(ch, nn.LayerNorm):
            r=RMSNorm(ch.normalized_shape[0], ch.eps); r.weight.data.copy_(ch.weight.data); setattr(m,name,r)
        else: swap_norm(ch)
def gelu_tanh(m):
    for mod in m.modules():
        if isinstance(mod, nn.GELU): mod.approximate='tanh'
ap=argparse.ArgumentParser(); ap.add_argument('--variant',default='base'); ap.add_argument('--size',default='400M'); ap.add_argument('--steps',type=int,default=2); a=ap.parse_args()
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
if a.size=='33M': cfg=SmallImageConfig(d_model=512,n_layer=10,n_head=8,ffn_dim=2048,total_tokens=256)
else: cfg=SmallImageConfig(d_model=1280,n_layer=16,n_head=16,ffn_dim=7168,total_tokens=256)
model=SmallARImageModel(cfg)
if 'gelu' in a.variant or a.variant=='all': gelu_tanh(model)
if 'rms' in a.variant or a.variant=='all': swap_norm(model)
opt=bnb.optim.AdamW8bit(model.parameters(),lr=1e-4)
x=torch.randint(0,1024,(1,256)); text=torch.zeros(1,768)
for _ in range(1):
    th,ge=model(x,text); loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
ts=[]
for _ in range(a.steps):
    t0=time.perf_counter(); th,ge=model(x,text)
    loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ts.append(time.perf_counter()-t0)
print(json.dumps(dict(size=a.size,variant=a.variant,steps=a.steps,step_times=ts,mean=sum(ts)/len(ts),loss=float(loss.detach())),ensure_ascii=False))