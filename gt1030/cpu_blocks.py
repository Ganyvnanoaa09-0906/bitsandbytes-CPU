import sys, torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen4bit_loader import load_transformer
torch.set_num_threads(6); torch.manual_seed(0)
S,T=64,8
hs=torch.randn(1,S,64); enc=torch.randn(1,T,4096); ts=torch.tensor([0.5]); shapes=[[(1,8,8)]]
mask=torch.cat([torch.zeros(1,T,dtype=torch.bool),torch.ones(1,S//4,dtype=torch.bool)],1)
m=load_transformer(); blocks=[b for n,b in m.named_modules() if n.startswith('transformer_blocks.') and n.count('.')==1]
out={}; hooks=[]
for i,b in enumerate(blocks):
    def mk(i):
        def h(mod,inp,o):
            out[i]=(o.detach() if torch.is_tensor(o) else o[0].detach())
        return h
    hooks.append(b.register_forward_hook(mk(i)))
with torch.no_grad(): m(hidden_states=hs,encoder_hidden_states=enc,timestep=ts,img_shapes=shapes,img_mask=mask,return_dict=False)
for h in hooks: h.remove()
torch.save({'hs':hs,'enc':enc,'ts':ts,'mask':mask,'blocks':out}, r'D:\work\research\bench_results\qwen_cpu_blocks.pt')
print('saved cpu blocks',len(out))
