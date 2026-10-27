# -*- coding: utf-8 -*-
"""修 selftest：torch 侧 backward 必须用带梯度的前向（no_grad 下的图不能反传）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi_selftest.py'
s = io.open(p, encoding='utf-8').read()
old = """    t0 = time.time()
    lo = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1))
    lo.backward()"""
new = """    t0 = time.time()
    lg2 = m.forward_logits(xt, None)[1][0]
    lo = F.cross_entropy(lg2[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1))
    lo.backward()"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched')
