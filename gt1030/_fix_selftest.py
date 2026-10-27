# -*- coding: utf-8 -*-
"""修 selftest：grad 用 named_parameters() 取（state_dict() 里是 detached 副本）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model_selftest.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""    print('--- 权重梯度（device fp16 vs torch fp32）---')""",
"""    P = dict(m.named_parameters())
    print('--- 权重梯度（device fp16 vs torch fp32）---')""")
s = s.replace("m.state_dict()[p + 'mlp.w3.weight'].grad.T.numpy()",
              "P[p + 'mlp.w3.weight'].grad.T.numpy()")
s = s.replace("m.state_dict()[p + 'mlp.w12.weight'].grad.T.numpy()",
              "P[p + 'mlp.w12.weight'].grad.T.numpy()")
s = s.replace("m.state_dict()[p + 'attn.o.weight'].grad.T.numpy()",
              "P[p + 'attn.o.weight'].grad.T.numpy()")
s = s.replace("gq = m.state_dict()[p + 'attn.q.weight'].grad.T.numpy()",
              "gq = P[p + 'attn.q.weight'].grad.T.numpy()")
s = s.replace("gk = m.state_dict()[p + 'attn.k.weight'].grad.T.numpy()",
              "gk = P[p + 'attn.k.weight'].grad.T.numpy()")
s = s.replace("gv = m.state_dict()[p + 'attn.v.weight'].grad.T.numpy()",
              "gv = P[p + 'attn.v.weight'].grad.T.numpy()")
s = s.replace("m.state_dict()['gen_head.lin.weight'].grad.T.numpy()",
              "P['gen_head.lin.weight'].grad.T.numpy()")
s = s.replace("m.state_dict()['token_emb.weight'].grad.numpy()",
              "P['token_emb.weight'].grad.numpy()")
s = s.replace("m.state_dict()['row_emb'].grad.numpy()", "P['row_emb'].grad.numpy()")
s = s.replace("m.state_dict()['col_emb'].grad.numpy()", "P['col_emb'].grad.numpy()")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched selftest')
