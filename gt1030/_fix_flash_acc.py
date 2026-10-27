# -*- coding: utf-8 -*-
"""修 flash fwd 的核心 bug：我拿 acc[4][4] 既当 S 的临时累加器、又当 O 的持久累加器了。
必须分开：sacc（每个列块重算 S）+ oacc（跨列块持续累加 O）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\flash_attn.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""    int myr0 = ly * 4, myc0 = lx * 4;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;""",
"""    int myr0 = ly * 4, myc0 = lx * 4;
    float oacc[4][4];          // O 的持久累加器（跨列块）
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) oacc[i][j] = 0.f;""")

s = s.replace("""        // S = Qt · KT  （每线程 4 行 × 4 列，k=64 全在 local）
        for (int kk = 0; kk < 64; ++kk) {""",
"""        // S = Qt · KT  （每线程 4 行 × 4 列，k=64 全在 local）
        float sacc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) sacc[i][j] = 0.f;
        for (int kk = 0; kk < 64; ++kk) {""")
s = s.replace("""                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }""",
"""                for (int j = 0; j < 4; ++j) sacc[i][j] += av[i] * bv[j];
        }""")
s = s.replace("""                float sv = acc[i][j] * scale;""", """                float sv = sacc[i][j] * scale;""")
s = s.replace("""                acc[i][j] *= rs;                            // O 累加器按行缩放""",
              """                oacc[i][j] *= rs;                           // O 累加器按行缩放""")
s = s.replace("""                acc[0][j] += pv[0] * a0; acc[1][j] += pv[1] * a0;
                acc[2][j] += pv[2] * a0; acc[3][j] += pv[3] * a0;""",
"""                oacc[0][j] += pv[0] * a0; oacc[1][j] += pv[1] * a0;
                oacc[2][j] += pv[2] * a0; oacc[3][j] += pv[3] * a0;""")
s = s.replace("""            Op[(long)r * 64 + c] = (half)(acc[i][j] * inv);""",
              """            Op[(long)r * 64 + c] = (half)(oacc[i][j] * inv);""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched: sacc / oacc 分离')
