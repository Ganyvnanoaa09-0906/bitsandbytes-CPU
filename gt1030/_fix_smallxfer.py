# -*- coding: utf-8 -*-
"""优化小向量的 host 往返：60 次 (down+up) 各自 finish 太贵，改成批量（每步 2 次同步）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""    def down(self, b, shape, dtype=np.float16):
        a = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.q, a, b)
        self.q.finish()
        return a""",
"""    def down(self, b, shape, dtype=np.float16, sync=True):
        a = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.q, a, b)
        if sync:
            self.q.finish()
        return a""")

s = s.replace("""    def _upload_small(self, k):
        d = self.dev
        v = self.host[k]
        buf = {'ln1': self.ln1, 'ln2': self.ln2, 'qw': self.qw, 'kw': self.kw,
               'ls1': self.ls1, 'ls2': self.ls2}[k]
        arr = v.astype(np.float16) if k in ('ln1', 'ln2') else v.astype(np.float32)
        cl.enqueue_copy(d.q, buf, np.ascontiguousarray(arr), is_blocking=True)
        d.q.finish()""",
"""    def _small_buf(self, k):
        return {'ln1': self.ln1, 'ln2': self.ln2, 'qw': self.qw, 'kw': self.kw,
                'ls1': self.ls1, 'ls2': self.ls2}[k]

    def _upload_small(self, k, sync=True):
        d = self.dev
        v = self.host[k]
        arr = v.astype(np.float16) if k in ('ln1', 'ln2') else v.astype(np.float32)
        cl.enqueue_copy(d.q, self._small_buf(k), np.ascontiguousarray(arr))
        if sync:
            d.q.finish()""")

s = s.replace("""        # ---- 小向量（ln/qk-norm/ls）：host AdamW，然后把新值传回 device ----
        for k, v in self.host.items():
            g = d.down(self.dw[k], (v.size,), np.float32)
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g
            v -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps) - lr * wd * v
            self._upload_small(k)""",
"""        # ---- 小向量（ln/qk-norm/ls）：host AdamW；批量下载/上传，只同步两次 ----
        grads = {k: d.down(self.dw[k], (v.size,), np.float32, sync=False)
                 for k, v in self.host.items()}
        d.finish()
        for k, v in self.host.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g
            v -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps) - lr * wd * v
            self._upload_small(k, sync=False)
        d.finish()""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched batched small-vector transfer')
