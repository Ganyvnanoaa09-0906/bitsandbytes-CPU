# -*- coding: utf-8 -*-
"""探针 11：用显式 copy 上传，并在每个 Param 创建后立即回读；再在整机构建后回读全部。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

WHICH = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # 0=COPY_HOST_PTR 1=显式copy


def up_copyhost(self, a):
    return cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                     hostbuf=np.ascontiguousarray(a))


def up_explicit(self, a):
    a = np.ascontiguousarray(a)
    b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(self.q, b, a)
    self.q.finish()
    return b


MM.Dev.up = up_explicit if WHICH else up_copyhost

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
checks = []
orig = MM.Param.__init__


def patched(self, dev, wf, lr_scale=1.0):
    orig(self, dev, wf, lr_scale)
    back = dev.down(self.w, wf.shape, np.float16)
    ok = np.allclose(back, wf, atol=1e-3)
    checks.append((wf.shape, ok))


MM.Param.__init__ = patched
model = MM.V2ModelDev(ckpt)
bad = [c for c in checks if not c[1]]
print('mode=%s  Params=%d  创建时回读失败=%d' % ('explicit' if WHICH else 'copyhost',
                                                 len(checks), len(bad)))
print('  失败样例:', bad[:3])
# 构建完成后，逐个回读全部权重
fails = 0
for ly in model.layers:
    for k, p in ly.W.items():
        a = model.dev.down(p.w, p.shape_of if hasattr(p, 'shape_of') else None) if False else None
print('（整机后回读见下一段）')
d = model.dev
for li in (0, 5, 9):
    a = d.down(model.layers[li].W['Wqkv'].w, (MM.MD, MM.WROW)).astype(np.float32)
    print('L%d Wqkv absmax=%.4f' % (li, np.abs(a).max()))
print('head absmax=%.4f' % np.abs(d.down(model.head.w, (MM.MD, model.V)).astype(np.float32)).max())
