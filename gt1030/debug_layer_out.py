import sys, numpy as np
sys.path.insert(0,r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model as FM
import gt1030.full_gpu_layer as L
rng=np.random.default_rng(0)
x=(rng.standard_normal((FM.M,FM.MD)).astype(np.float32)*0.1).astype(np.float16)
xdev=FM.upload(x)
layers=[FM.Layer(rng,i) for i in range(3)]
outs=[]; cur=xdev
for i,l in enumerate(layers):
    cur=l.forward(cur); L.A.queue.finish()
    o=np.empty((FM.M,FM.MD),dtype=np.float16); L.cl.enqueue_copy(L.A.queue,o,cur); L.A.queue.finish()
    outs.append(o.copy()); print('layer',i,'sum',float(o.astype(np.float32).sum()),'max',float(o.max()), flush=True)
print('eq 0-1',np.array_equal(outs[0],outs[1]),'eq1-2',np.array_equal(outs[1],outs[2]))
print('diff01 max',float(np.abs(outs[0].astype(np.float32)-outs[1]).max()),'diff12',float(np.abs(outs[1].astype(np.float32)-outs[2]).max()))
