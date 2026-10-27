# -*- coding: utf-8 -*-
"""补 Head.opt_step / Head.grad_norm。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()
old = """    def backward(self):
        d = self.dev
        d.activate()
        L.hgemm_tn(self._xb, self.dlogits, self.W.grad, MD, self.V, M)"""
new = """    def grad_norm(self):
        d = self.dev
        d.activate()
        g = d.down(self.W.grad, (self.W.n,), np.float16).astype(np.float32)
        return float((g * g).sum())

    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.dev.activate()
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        self.W.step(lr, b1, b2, eps, wd, bc1, bc2)

    def backward(self):
        d = self.dev
        d.activate()
        L.hgemm_tn(self._xb, self.dlogits, self.W.grad, MD, self.V, M)"""
assert old in s
s = s.replace(old, new, 1)

# Model.grad_norm 里那行 hasattr 清理掉
s = s.replace("""        tot = self.head.dev_gnorm() if hasattr(self.head, 'dev_gnorm') else 0.0
        d = self.head.dev
        g = d.down(self.head.W.grad, (self.head.W.n,), np.float16).astype(np.float32)
        tot += float((g * g).sum())""",
"""        tot = self.head.grad_norm()""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched Head.opt_step')
