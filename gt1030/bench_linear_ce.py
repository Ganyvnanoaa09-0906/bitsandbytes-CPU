import sys, time, psutil, torch
sys.path.insert(0,r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes.fused as F
torch.set_num_threads(6)
M,D,V=256,2048,151936
h=torch.randn(M,D); w=torch.randn(V,D); lab=torch.randint(0,V,(M,))
def rss(): return psutil.Process().memory_info().rss/1e9
base=rss()
t=time.perf_counter(); logits=h@w.T; full_mem=rss(); loss=torch.nn.functional.cross_entropy(logits,lab); full_t=time.perf_counter()-t; del logits
torch.cuda.empty_cache() if torch.cuda.is_available() else None
t=time.perf_counter(); loss2=F.linear_cross_entropy(h,w,lab,chunk=1024); chunk_t=time.perf_counter()-t; chunk_mem=rss()
print('full_ms',round(full_t*1e3,1),'full_rss_inc',round(full_mem-base,3),'chunk_ms',round(chunk_t*1e3,1),'chunk_rss_inc',round(chunk_mem-base,3),'rel',float(abs(loss-loss2)/abs(loss)))
