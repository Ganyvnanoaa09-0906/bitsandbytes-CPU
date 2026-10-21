import sys, gc, numpy as np
sys.path.insert(0,r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model as FM
import gt1030.full_gpu_layer as L
rng=np.random.default_rng(0)
x=(rng.standard_normal((FM.M,FM.MD)).astype(np.float32)*0.1).astype(np.float16)
xdev=FM.upload(x)
lay=FM.Layer(rng,0)
out=lay.forward(xdev); L.A.queue.finish()
def dl():
    a=np.empty((FM.M,FM.MD),dtype=np.float16); L.cl.enqueue_copy(L.A.queue,a,out); L.A.queue.finish(); return a
before=dl(); print('before sum',float(before.astype(np.float32).sum()), flush=True)
gc.collect(); L.A.queue.finish(); after_gc=dl(); print('after gc diff',float(np.abs(after_gc.astype(np.float32)-before).max()), flush=True)
dummies=[]
for i in range(100):
    dummies.append(L.cl.Buffer(L.A.ctx, L.A.mf.READ_WRITE, (FM.M*FM.MD*2)))
L.A.queue.finish(); after_alloc=dl(); print('after 100 buffers diff',float(np.abs(after_alloc.astype(np.float32)-before).max()), flush=True)
# also allocate many small weight-sized buffers + host copies
for i in range(100):
    dummies.append(L.cl.Buffer(L.A.ctx, L.A.mf.READ_WRITE|L.A.mf.COPY_HOST_PTR, hostbuf=np.zeros((FM.MD,FM.FF),dtype=np.float16)))
L.A.queue.finish(); after_alloc2=dl(); print('after 200 buffers diff',float(np.abs(after_alloc2.astype(np.float32)-before).max()), flush=True)
