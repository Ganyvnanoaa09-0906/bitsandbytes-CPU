import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os,sys,time,torch
sys.path.insert(0,r'D:\work\bitsandbytes-CPU')
import wan_vae_profile as P
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanRMS_norm
torch.set_num_threads(6)
model=P.load_decoder(); latent=torch.randn(1,16,5,8,8)
with torch.no_grad():
 t=time.perf_counter(); out0=model.decode(latent,return_dict=False)[0]; base=time.perf_counter()-t
def fast(self,x):
 ss=x.pow(2).sum(dim=1,keepdim=True); return (x*torch.rsqrt(ss.clamp_min(1e-24)))*self.scale*self.gamma+self.bias
for m in model.decoder.modules():
 if isinstance(m,WanCausalConv3d): m.to(memory_format=torch.channels_last_3d)
WanRMS_norm.forward=fast
x=latent.contiguous(memory_format=torch.channels_last_3d)
with torch.no_grad():
 t=time.perf_counter(); out1=model.decode(x,return_dict=False)[0]; comb=time.perf_counter()-t
print('baseline %.1fms combined %.1fms speedup %.2fx maxdiff %.2e'%(base*1e3,comb*1e3,base/comb,float((out0-out1).abs().max())))
