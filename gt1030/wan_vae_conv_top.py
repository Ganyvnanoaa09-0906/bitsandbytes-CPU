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
import os,sys,time,torch
sys.path.insert(0,r'D:\work\bitsandbytes-CPU')
from scripts.wan import wan_vae_profile as P
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
torch.set_num_threads(6)
model=P.load_decoder(); dec=model.decoder
rec={}
def mk(name,m):
 def pre(mod,args): m._t0=time.perf_counter(); m._sh=tuple(args[0].shape) if args else ()
 def post(mod,args,out): rec[name]=(time.perf_counter()-m._t0,m._sh,tuple(mod.weight.shape),tuple(mod.stride),tuple(mod.padding))
 return m.register_forward_pre_hook(pre),m.register_forward_hook(post)
hs=[]
for n,m in dec.named_modules():
 if isinstance(m,WanCausalConv3d): hs.extend(mk(n,m))
lat=torch.randn(1,16,5,8,8)
with torch.no_grad(): model.decode(lat,return_dict=False)
rec.clear()
with torch.no_grad(): t=time.perf_counter(); model.decode(lat,return_dict=False); total=time.perf_counter()-t
print('total %.1fms'%(total*1e3))
for n,(dt,sh,w,st,p) in sorted(rec.items(),key=lambda kv:-kv[1][0])[:12]:
 print('%7.1fms %-40s in=%s w=%s s%s p%s'%(dt*1e3,n,sh,w,st,p))
