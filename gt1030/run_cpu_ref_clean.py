import sys, torch, time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen4bit_loader import load_transformer
torch.set_num_threads(6)
d=torch.load(r'D:\work\research\bench_results\qwen_hybrid_out.pt', map_location='cpu')
m=load_transformer()
with torch.no_grad():
    t0=time.perf_counter(); y=m(hidden_states=d['hs'],encoder_hidden_states=d['enc'],timestep=d['ts'],img_shapes=[[(1,8,8)]],img_mask=d['mask'],return_dict=False)[0]; print('cpu',time.perf_counter()-t0)
rel=float((d['y']-y).norm()/y.norm()); print('rel',rel,'std hybrid',float(d['y'].std()),'cpu',float(y.std()))
