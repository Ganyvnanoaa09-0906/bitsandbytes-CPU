# -*- coding: utf-8 -*-
"""让 ar_v2_model 支持任意层数（默认还是 10），方便做小规模的稳定性/对拍实验。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()
old = """        assert cfg['n_layer'] == 10
        sd = ck['model']
        self.ck = ck
        self.cfg = cfg
        self.dev = dev or Dev()
        d = self.dev
        self.layers = [Layer(d, sd, i) for i in range(10)]"""
new = """        sd = ck['model']
        self.n_layer = int(cfg['n_layer'])
        self.ck = ck
        self.cfg = cfg
        self.dev = dev or Dev()
        d = self.dev
        self.layers = [Layer(d, sd, i) for i in range(self.n_layer)]"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched: variable layer count')
