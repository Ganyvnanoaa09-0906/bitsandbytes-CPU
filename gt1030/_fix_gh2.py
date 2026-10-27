# -*- coding: utf-8 -*-
"""gemm_half_v2: 通用 TM/TN 向量读（TM/TN 为 4 的倍数）+ 更多大 MS 配置。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\gemm_half_v2.py'
s = io.open(p, encoding='utf-8').read()

start = s.index('            float av[TM], bv[TN];')
end = s.index('#pragma unroll\n            for (int i = 0; i < TM; ++i)')
new = '''            float av[TM], bv[TN];
#pragma unroll
            for (int t = 0; t < TM; t += 4) {
                half4 a = vload4(0, (__local half*)At + kk * MS + m0 + t);
                av[t] = (float)a.x; av[t+1] = (float)a.y;
                av[t+2] = (float)a.z; av[t+3] = (float)a.w;
            }
#pragma unroll
            for (int t = 0; t < TN; t += 4) {
                half4 b = vload4(0, (__local half*)Bl + kk * NS + n0 + t);
                bv[t] = (float)b.x; bv[t+1] = (float)b.y;
                bv[t+2] = (float)b.z; bv[t+3] = (float)b.w;
            }
'''
s = s[:start] + new + s[end:]

anchor = "    dict(name='256x64 t16x4 BK16', MS=256, NS=64, TM=16, TN=4, LX=16, LY=16, BK=16),\n"
assert anchor in s
extra = ("    dict(name='256x64 t16x4 BK32', MS=256, NS=64, TM=16, TN=4, LX=16, LY=16, BK=32),\n"
         "    dict(name='256x128 t16x8 BK16', MS=256, NS=128, TM=16, TN=8, LX=16, LY=16, BK=16),\n"
         "    dict(name='512x64 t32x4 BK16', MS=512, NS=64, TM=32, TN=4, LX=16, LY=16, BK=16),\n")
s = s.replace(anchor, anchor + extra)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
