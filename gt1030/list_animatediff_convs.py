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
import os,sys,json,torch
sys.path.insert(0,r'D:\work\video_mvp'); sys.path.insert(0,r'D:\work\bitsandbytes-CPU'); sys.path.insert(0,r'D:\work\bitsandbytes-CPU\bitsandbytes'); sys.path.insert(1,r'D:\work\bitsandbytes-CPU\pytorch')
os.chdir(r'D:\work\video_mvp')
from scripts.lib.animatediff_train import load_animatediff_model, resolve_video_model
UNET_DIR=r'D:\work\textmodel\sd15_base\unet'; ADAPTER_DIR=r'D:\work\textmodel\animatediff-motion-adapter-v1-5-2'
unet_dir,adapter_dir=resolve_video_model('%s,%s'%(UNET_DIR,ADAPTER_DIR))
model=load_animatediff_model(unet_dir,adapter_dir).eval()
recs={}
def hook(name,m,inp,out):
    x=inp[0]; recs[name]=tuple(x.shape); return None
hs=[m.register_forward_hook(lambda m,i,o,n=n: hook(n,m,i,o)) for n,m in model.named_modules() if isinstance(m,torch.nn.Conv2d)]
B,T=1,16
x=torch.randn(B,4,T,16,16); t=torch.randint(0,1000,(B,)); ctx=torch.randn(B*T,77,768)
with torch.no_grad(): model(x,t,encoder_hidden_states=ctx)
rows=[]
for name,m in model.named_modules():
    if isinstance(m,torch.nn.Conv2d) and name in recs:
        sh=recs[name]; N,C,H,W=sh; fl=2*N*H*W*m.out_channels*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        rows.append((fl,name,sh,m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding))
rows.sort(reverse=True)
for fl,name,sh,ci,co,k,s,p in rows[:8]:
    print('%9.2f GF %-45s input=%s in%4d out%4d k%s s%s p%s'%(fl/1e9,name,sh,ci,co,k,s,p))
json.dump([dict(flops=fl,name=name,input=list(sh),in_ch=ci,out_ch=co,k=list(k),stride=list(s),padding=list(p)) for fl,name,sh,ci,co,k,s,p in rows],open(r'D:\work\research\bench_results\video_convs_shapes.json','w'),indent=2)
