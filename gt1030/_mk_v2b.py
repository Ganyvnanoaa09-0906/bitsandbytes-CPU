# -*- coding: utf-8 -*-
"""再扫一轮：更大的 BK（减少屏障/载入次数）+ 更好的分块组合，看能否把 GEMM 从 500~700 GFLOPS 抬上去。
和 CLBlast 在同一进程/同一时刻对拍，所以系统负载漂移不影响"比值"这个结论。"""
import io
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

# 复用 gemm_half_v2 的模板与 bench 逻辑，只换配置表
src = io.open(r'D:\work\bitsandbytes-CPU\gt1030\gemm_half_v2.py', encoding='utf-8').read()
i = src.index('CFGS = [')
j = src.index(']\n', i) + 2
newcfg = '''CFGS = [
    dict(name='256x64 t16x4 BK32', MS=256, NS=64, TM=16, TN=4, LX=16, LY=16, BK=32),
    dict(name='256x64 t16x4 BK64', MS=256, NS=64, TM=16, TN=4, LX=16, LY=16, BK=64),
    dict(name='256x64 t16x4 BK128', MS=256, NS=64, TM=16, TN=4, LX=16, LY=16, BK=128),
    dict(name='128x64 t8x4 BK64', MS=128, NS=64, TM=8, TN=4, LX=16, LY=16, BK=64),
    dict(name='128x64 t8x4 BK128', MS=128, NS=64, TM=8, TN=4, LX=16, LY=16, BK=128),
    dict(name='256x128 t16x8 BK64', MS=256, NS=128, TM=16, TN=8, LX=16, LY=16, BK=64),
    dict(name='128x128 t8x8 BK64', MS=128, NS=128, TM=8, TN=8, LX=16, LY=16, BK=64),
    dict(name='256x32 t16x2 BK64', MS=256, NS=32, TM=16, TN=2, LX=16, LY=16, BK=64),
]
'''
src = src[:i] + newcfg + src[j:]
io.open(r'D:\work\bitsandbytes-CPU\gt1030\gemm_half_v2b.py', 'w', encoding='utf-8',
        newline='\n').write(src)
print('wrote gemm_half_v2b.py')
