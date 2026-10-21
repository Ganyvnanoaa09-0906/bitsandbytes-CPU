import sys, numpy as np
sys.path.insert(0,r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model as FM
import gt1030.full_gpu_layer as L
rng=np.random.default_rng(0)
x=(rng.standard_normal((FM.M,FM.MD)).astype(np.float32)*0.1).astype(np.float16)
xdev=FM.upload(x)
layers=[FM.Layer(rng,i) for i in range(3)]
for i,l in enumerate(layers):
    print('layer',i,'x2 ptr',int(l.b['x2'].int_ptr),'x1 ptr',int(l.b['x1'].int_ptr),'h1 ptr',int(l.b['h1'].int_ptr))
print('xdev ptr',int(xdev.int_ptr))
cur=xdev
for i,l in enumerate(layers):
    cur=l.forward(cur); L.A.queue.finish()
    print('after layer',i,'cur ptr',int(cur.int_ptr),'same as layer x2?',cur.int_ptr==l.b['x2'].int_ptr)
