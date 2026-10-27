# -*- coding: utf-8 -*-
"""补 causal 掩码：对角块内部 c > r 的元素必须排除（max 里当 -inf、P 里当 0）。
这是 rel=0.99 的真凶 —— 之前只在"整块"层面按列块截断，块内没掩码。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\flash_attn.py'
s = io.open(p, encoding='utf-8').read()

old1 = """        float pm[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            pm[i] = -1e30f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float sv = sacc[i][j] * scale;
                St[(myr0 + i) * 64 + myc0 + j] = (half)sv;
                pm[i] = fmax(pm[i], sv);
            }
        }"""
new1 = """        float pm[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int grow = r0 + myr0 + i;
            pm[i] = -1e30f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gcol = c0 + myc0 + j;
                float sv = sacc[i][j] * scale;
                int keep = (grow >= gcol) ? 1 : 0;          // causal 掩码
                St[(myr0 + i) * 64 + myc0 + j] = (half)(keep ? sv : -30000.f);
                pm[i] = fmax(pm[i], keep ? sv : -1e30f);
            }
        }"""
assert old1 in s
s = s.replace(old1, new1)

old2 = """#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float e = exp((float)St[(myr0 + i) * 64 + myc0 + j] - m);
                St[(myr0 + i) * 64 + myc0 + j] = (half)e;   // 就地变成 P
                sum += e;
                oacc[i][j] *= rs;                           // O 累加器按行缩放
            }"""
new2 = """            int grow = r0 + myr0 + i;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gcol = c0 + myc0 + j;
                float e = 0.f;
                if (grow >= gcol) e = exp((float)St[(myr0 + i) * 64 + myc0 + j] - m);
                St[(myr0 + i) * 64 + myc0 + j] = (half)e;   // 就地变成 P
                sum += e;
                oacc[i][j] *= rs;                           // O 累加器按行缩放
            }"""
assert old2 in s
s = s.replace(old2, new2)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('causal 掩码已补')
