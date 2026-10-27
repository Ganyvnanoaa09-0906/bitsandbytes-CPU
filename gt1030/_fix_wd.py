# -*- coding: utf-8 -*-
"""修一个训练语义不一致：CPU trainer 用的是 bnb.optim.AdamW8bit 默认 weight_decay=0.01
（torch.optim.AdamW 默认也是 0.01），而我这边一直传 wd=0。短程（10 步）看不出来
（每步只差 lr*wd=5e-7），但几百步以后会有系统性差异。改成默认 0.01。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("""    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, clip=1.0):""",
"""    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, clip=1.0):""")
s = s.replace("""    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.dev.activate()""",
"""    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.dev.activate()""")
s = s.replace("""    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, max_norm=1.0):""",
"""    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, max_norm=1.0):""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)

# selftest 里也要用 0.01（和 torch.optim.AdamW 默认一致）
p2 = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi_selftest.py'
s2 = io.open(p2, encoding='utf-8').read()
s2 = s2.replace("""        model.opt_step(args.lr, st)""", """        model.opt_step(args.lr, st, wd=0.01)""")
io.open(p2, 'w', encoding='utf-8', newline='\n').write(s2)
print('patched wd=0.01')
