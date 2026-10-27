# -*- coding: utf-8 -*-
"""Head 也补 opt_dev 别名（它的 opt_step 本来就只有 Param.step，无同步）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()
old = """    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.dev.activate()
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        self.W.step(lr, b1, b2, eps, wd, bc1, bc2)"""
new = """    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.dev.activate()
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        self.W.step(lr, b1, b2, eps, wd, bc1, bc2)

    def opt_dev(self, lr, b1, b2, eps, wd, bc1, bc2):
        self.dev.activate()
        self.W.step(lr, b1, b2, eps, wd, bc1, bc2)"""
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('Head.opt_dev added')
