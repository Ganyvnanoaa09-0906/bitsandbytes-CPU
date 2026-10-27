# -*- coding: utf-8 -*-
"""让 ar_v2_model 支持 B=1（M=1024）：V2 里 M 是硬编码 2048，导入 L 之前改掉。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()
old = """import gt1030.full_gpu_layer as L          # noqa: E402  (import 顺序不能动：它读 A.B/A.T)
import gt1030.full_gpu_step_v2 as V2       # noqa: E402"""
new = """import gt1030.full_gpu_step_v2 as V2       # noqa: E402
V2.M = BATCH * SEQ                        # V2 里 M 硬编码 2048；B=1 时要用 1024
import gt1030.full_gpu_layer as L          # noqa: E402  (import 顺序不能动：它读 A.B/A.T/V2.M)"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched M derivation')
