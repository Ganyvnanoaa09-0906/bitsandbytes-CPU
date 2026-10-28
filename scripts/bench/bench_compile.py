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
import sys, time, argparse, json, torch, torch.nn.functional
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes
ap=argparse.ArgumentParser(); ap.add_argument('--size',default='33M'); ap.add_argument('--steps',type=int,default=3); a=ap.parse_args()
torch.set_num_threads(6); torch.backends.mkldnn.enabled=True
if a.size=='33M':
    cfg=SmallImageConfig(d_model=512,n_layer=10,n_head=8,ffn_dim=2048,total_tokens=256)
else:
    cfg=SmallImageConfig(d_model=1280,n_layer=16,n_head=16,ffn_dim=7168,total_tokens=256)
model=SmallARImageModel(cfg); opt=bnb.optim.AdamW8bit(model.parameters(),lr=1e-4)
x=torch.randint(0,1024,(1,256)); text=torch.zeros(1,768)
def run_step():
    th,ge=model(x,text)
    loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
# baseline
for _ in range(2): run_step()
t0=time.perf_counter(); [run_step() for _ in range(a.steps)]; base=(time.perf_counter()-t0)/a.steps
# compile
try:
    cm=torch.compile(model, mode='default')
    # warmup separately
    t0=time.perf_counter()
    for _ in range(2):
        th,ge=cm(x,text); loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    warm=time.perf_counter()-t0
    t0=time.perf_counter()
    for _ in range(a.steps):
        th,ge=cm(x,text); loss=F.cross_entropy(ge.reshape(-1,1024),x.reshape(-1))+0.1*F.cross_entropy(th.reshape(-1,1024),x[:,:16].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    comp=(time.perf_counter()-t0)/a.steps
    print(json.dumps(dict(size=a.size,baseline_s=base,compiled_s=comp,warmup_s=warm,speedup=base/comp),ensure_ascii=False))
except Exception as e:
    print(json.dumps(dict(size=a.size,baseline_s=base,error=f'{type(e).__name__}: {str(e)[:300]}'),ensure_ascii=False))