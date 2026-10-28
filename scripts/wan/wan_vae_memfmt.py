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
import os,sys,time,torch
sys.path.insert(0,r'D:\work\bitsandbytes-CPU')
import wan_vae_profile as P
torch.set_num_threads(6)
model=P.load_decoder(); latent=torch.randn(1,16,5,8,8)
with torch.no_grad():
 t=time.perf_counter(); out0=model.decode(latent,return_dict=False)[0]; base=time.perf_counter()-t
print('baseline %.1fms'%(base*1e3))
try:
 model.decoder=model.decoder.to(memory_format=torch.channels_last_3d)
 with torch.no_grad():
  t=time.perf_counter(); out1=model.decode(latent,return_dict=False)[0]; cl=time.perf_counter()-t
 print('channels_last_3d %.1fms speedup %.2fx maxdiff %.2e'%(cl*1e3,base/cl,float((out0-out1).abs().max())))
except Exception as e:
 print('channels_last_3d failed:',type(e).__name__,str(e)[:300])
