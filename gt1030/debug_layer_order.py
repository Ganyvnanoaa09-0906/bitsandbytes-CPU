import sys, numpy as np
sys.path.insert(0,r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model as FM
import gt1030.full_gpu_layer as L
rng=np.random.default_rng(0)
layers=[FM.Layer(rng,0),FM.Layer(rng,1)]
x=(rng.standard_normal((FM.M,FM.MD)).astype(np.float32)*0.1).astype(np.float16)
xdev=FM.upload(x)
cur=xdev
for i,lay in enumerate(layers):
    cur=lay.forward(cur); L.A.queue.finish()
    out=np.empty((FM.M,FM.MD),dtype=np.float16); L.cl.enqueue_copy(L.A.queue,out,cur); L.A.queue.finish()
    print('layer',i,'max',float(np.abs(out).max()),'mean',float(out.mean()), flush=True)
