# -*- coding: utf-8 -*-
"""flash_attn.py 测试脚本改用 ar_v2_multi 的 Dev/program（softmax_causal 在那里）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\flash_attn.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("""    import gt1030.ar_v2_multi as MM
    import gt1030.attention_gpu as A

    ctx, q = A.ctx, A.queue""",
"""    import gt1030.ar_v2_multi as MM
    import gt1030.attention_gpu as A
    d0 = MM.Dev()
    d0.activate()

    ctx, q = d0.ctx, d0.q""")
s = s.replace("A.K['softmax_causal']", "d0.K['softmax_causal']")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('fixed')
