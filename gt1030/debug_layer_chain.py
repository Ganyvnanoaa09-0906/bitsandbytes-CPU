import sys, numpy as np, torch
sys.path.insert(0,r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model as FM
import gt1030.full_gpu_layer as L
from gt1030.debug_full_gpu_model import cpu_layer
rng=np.random.default_rng(0)
x=(rng.standard_normal((FM.M,FM.MD)).astype(np.float32)*0.1).astype(np.float16)
xdev=FM.upload(x)
layers=[FM.Layer(rng,i) for i in range(10)]
L.A.queue.finish()  # after construction
def dl(b):
    a=np.empty((FM.M,FM.MD),dtype=np.float16); L.cl.enqueue_copy(L.A.queue,a,b); L.A.queue.finish(); return a
def gpu_layer(lay, xin):
    out = lay.forward(xin); L.A.queue.finish(); return out
def cpu_one(lay, xt):
    return cpu_layer(xt, torch.from_numpy(lay.WQKV['host'].astype(np.float32)),
                     torch.from_numpy(lay.WO['host'].astype(np.float32)),
                     torch.from_numpy(lay.W1['host'].astype(np.float32)),
                     torch.from_numpy(lay.W3['host'].astype(np.float32)),
                     torch.from_numpy(lay.W2['host'].astype(np.float32)))
cur=xdev
with torch.no_grad():
    for i,lay in enumerate(layers):
        inp=dl(cur)
        outbuf=gpu_layer(lay,cur)
        out=dl(outbuf)
        ref=cpu_one(lay, torch.from_numpy(inp.astype(np.float32)))
        err=float(np.abs(out.astype(np.float32)-ref.numpy()).max())
        print('layer',i,'in_sum',float(inp.astype(np.float32).sum()),'out_sum',float(out.astype(np.float32).sum()),'ref_sum',float(ref.sum()),'maxerr',err, flush=True)
        cur=outbuf
