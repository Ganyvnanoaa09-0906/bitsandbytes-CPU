import sys, time, statistics, torch
sys.path.insert(0,r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes.fused as F
torch.set_num_threads(6)
def med(fn,n=5):
 fn(); ts=[]
 for _ in range(n):
  t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
 return statistics.median(ts)*1e3
for shape in [(2048,4096),(512,8192)]:
 x=torch.randn(*shape); w=torch.randn(shape[-1])
 t=med(lambda:F.rms_norm(x,w)); n=med(lambda:F.rms_norm_native(x,w))
 print('rms',shape,'torch',round(t,2),'native',round(n,2),'speedup',round(t/n,2))
g=torch.randn(2048,11008); u=torch.randn_like(g)
t=med(lambda:F.silu_mul(g,u)); n=med(lambda:F.silu_mul_native(g,u))
print('silu',tuple(g.shape),'torch',round(t,2),'native',round(n,2),'speedup',round(t/n,2))
