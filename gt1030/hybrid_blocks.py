import sys, torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen4bit_loader import load_transformer
from qwen4bit_igpu_linear_fp32 import to_igpu_f32
torch.set_num_threads(6); torch.manual_seed(0)
d=torch.load(r'D:\work\research\bench_results\qwen_cpu_blocks.pt', map_location='cpu')
m=load_transformer(); to_igpu_f32(m)
blocks=[b for n,b in m.named_modules() if n.startswith('transformer_blocks.') and n.count('.')==1]
out={}; hooks=[]
for i,b in enumerate(blocks):
    def mk(i):
        def h(mod,inp,o): out[i]=(o.detach() if torch.is_tensor(o) else o[0].detach())
        return h
    hooks.append(b.register_forward_hook(mk(i)))
with torch.no_grad(): m(hidden_states=d['hs'],encoder_hidden_states=d['enc'],timestep=d['ts'],img_shapes=[[(1,8,8)]],img_mask=d['mask'],return_dict=False)
for h in hooks: h.remove()
for i in d['blocks']:
    r=d['blocks'][i]; g=out[i]; rel=float((g-r).norm()/(r.norm()+1e-9))
    print('block %2d rel %.5f std %.2f vs %.2f'%(i,rel,float(g.std()),float(r.std())))
