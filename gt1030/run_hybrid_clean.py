import sys, torch, time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen4bit_loader import load_transformer
from qwen4bit_igpu_linear_fp32 import to_igpu_f32
torch.set_num_threads(6); torch.manual_seed(0)
S,T=64,8
hs=torch.randn(1,S,64); enc=torch.randn(1,T,4096); ts=torch.tensor([0.5]); shapes=[[(1,8,8)]]
mask=torch.cat([torch.zeros(1,T,dtype=torch.bool),torch.ones(1,S//4,dtype=torch.bool)],1)
m=load_transformer(); to_igpu_f32(m)
with torch.no_grad():
    t0=time.perf_counter(); y=m(hidden_states=hs,encoder_hidden_states=enc,timestep=ts,img_shapes=shapes,img_mask=mask,return_dict=False)[0]; print('hybrid',time.perf_counter()-t0)
torch.save({'hs':hs,'enc':enc,'ts':ts,'mask':mask,'y':y}, r'D:\work\research\bench_results\qwen_hybrid_out.pt')
print('saved')
