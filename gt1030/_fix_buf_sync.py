# -*- coding: utf-8 -*-
"""修 Dev.buf：每次原始分配后也 finish（探针 18 里『每次分配都插一次同步』时 3/3 正常）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()
old = """    def buf(self, nbytes):
        return cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)"""
new = """    def buf(self, nbytes):
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)
        self.q.finish()      # 连续分配不 sync 时本机驱动会坏（见 _t_mem_probe18）
        return b"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched Dev.buf with finish')
