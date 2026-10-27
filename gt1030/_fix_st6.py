# -*- coding: utf-8 -*-
"""selftest 里加上小向量梯度（ln1/ln2/qw/kw/ls1/ls2）的对照。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi_selftest.py'
s = io.open(p, encoding='utf-8').read()
old = """    cmp('head dW', model.head.dev.down(model.head.W.grad, (MM.MD, model.head.V), np.float16),
        P['gen_head.lin.weight'].grad.T.contiguous().numpy())"""
new = """    print('--- 小向量梯度（ln / qk-norm / ls）---')
    for li in inds:
        blk = model.blocks[li]
        p = 'blocks.%d.' % li
        for key, sd_name in (('ln1', 'ln1.w'), ('ln2', 'ln2.w'), ('qw', 'attn.q_norm.w'),
                             ('kw', 'attn.k_norm.w'), ('ls1', 'ls1'), ('ls2', 'ls2')):
            cmp('L%d d%s' % (li, key), blk.dev.down(blk.dw[key], (blk.host[key].size,), np.float32),
                P[p + sd_name].grad.numpy())
    cmp('head dW', model.head.dev.down(model.head.W.grad, (MM.MD, model.head.V), np.float16),
        P['gen_head.lin.weight'].grad.T.contiguous().numpy())"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched selftest with small-vector grads')
