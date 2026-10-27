# -*- coding: utf-8 -*-
"""探针 16：先吃掉 600MB / 1.2GB host 内存，再构建整机，看队列是否还活着（主机内存压力假说）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import psutil
import gt1030.ar_v2_model as MM

HOG_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
hog = np.ones(int(HOG_GB * 1e9 / 8), np.float64) if HOG_GB > 0 else None
vm = psutil.virtual_memory()
print('吃 %.1fGB 后 host avail=%.2fGB' % (HOG_GB, vm.available / 1e9))

d = MM.Dev()
q = d.q


def alive(tag):
    a = np.full(512, 3.0, np.float16)
    b = cl.Buffer(d.ctx, d.mf.READ_WRITE, 1024)
    cl.enqueue_copy(q, b, a, is_blocking=True)
    out = np.empty(512, np.float16)
    cl.enqueue_copy(q, out, b)
    q.finish()
    v = float(out[0])
    print('%-20s alive=%-5s value=%.2f  host_avail=%.2fGB'
          % (tag, v == 3.0, v, psutil.virtual_memory().available / 1e9))
    return v == 3.0


model = MM.V2ModelDev(r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt', dev=d)
alive('after model build')
p = model.layers[0].W['Wqkv']
a = d.down(p.w, (MM.MD, MM.WROW)).astype(np.float32)
print('L0.Wqkv absmax=%.4f (期望 0.1143)' % np.abs(a).max())
if HOG_GB > 0:
    del hog
