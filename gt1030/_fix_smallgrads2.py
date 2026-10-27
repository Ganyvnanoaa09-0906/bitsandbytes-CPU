# -*- coding: utf-8 -*-
"""Block 里补：小向量的 host AdamW + 上传 + grad-norm 贡献。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# ---- _upload_small + opt_step 扩展 ----
s = s.replace("""    def grad_norm(self):
        d = self.dev
        d.activate()
        tot = 0.0
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            p = self.W[k]
            g = d.down(p.grad, (p.n,), np.float16).astype(np.float32)
            tot += float((g * g).sum())
        return tot""",
"""    def _upload_small(self, k):
        d = self.dev
        v = self.host[k]
        buf = {'ln1': self.ln1, 'ln2': self.ln2, 'qw': self.qw, 'kw': self.kw,
               'ls1': self.ls1, 'ls2': self.ls2}[k]
        arr = v.astype(np.float16) if k in ('ln1', 'ln2') else v.astype(np.float32)
        cl.enqueue_copy(d.q, buf, np.ascontiguousarray(arr), is_blocking=True)
        d.q.finish()

    def grad_norm(self):
        d = self.dev
        d.activate()
        tot = 0.0
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            p = self.W[k]
            g = d.down(p.grad, (p.n,), np.float16).astype(np.float32)
            tot += float((g * g).sum())
        for k, buf in self.dw.items():
            g = d.down(buf, (len(self.host[k]),), np.float32)
            tot += float((g * g).sum())
        return tot""")

s = s.replace("""        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            self.W[k].step(lr, b1, b2, eps, wd, bc1, bc2)""",
"""        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            self.W[k].step(lr, b1, b2, eps, wd, bc1, bc2)
        # ---- 小向量（ln/qk-norm/ls）：host AdamW，然后把新值传回 device ----
        for k, v in self.host.items():
            g = d.down(self.dw[k], (v.size,), np.float32)
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g
            v -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps) - lr * wd * v
            self._upload_small(k)""")

# ---- clip 时也要缩放小向量梯度（它们已经是 float32 buffer）----
s = s.replace("""            for k in ('Wqkv', 'Wo', 'W12', 'W3'):
                p = blk.W[k]
                d.K['scale_half'](d.q, (p.n,), None, p.grad, np.float32(f), np.int32(p.n))
            d.finish()""",
"""            for k in ('Wqkv', 'Wo', 'W12', 'W3'):
                p = blk.W[k]
                d.K['scale_half'](d.q, (p.n,), None, p.grad, np.float32(f), np.int32(p.n))
            for k, buf in blk.dw.items():
                d.K['scale_float'](d.q, (blk.host[k].size,), None, buf, np.float32(f),
                                   np.int32(blk.host[k].size))
            d.finish()""")

# ---- 新 kernel scale_float ----
s = s.replace("""__kernel void scale_half(__global half* g, float f, int n) {""",
"""__kernel void scale_float(__global float* g, float f, int n) {
    int i = get_global_id(0); if (i >= n) return;
    g[i] = g[i] * f;
}
__kernel void scale_half(__global half* g, float f, int n) {""")
s = s.replace("""          'ce_softmax', 'scale_half', 'rms_dw_from_ss', 'col_sum_prod')""",
              """          'ce_softmax', 'scale_half', 'scale_float', 'rms_dw_from_ss', 'col_sum_prod')""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched opt_step + clip for small vectors')
