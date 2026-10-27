# -*- coding: utf-8 -*-
"""清掉 flash_attn.py 里残留的占位读（vload4 + (void)），它们是写代码时的脚手架。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\flash_attn.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("""            float av[4], bv[4];
            half4 a = vload4(0, (__local half*)Qt + (myr0 + 0) * 64 + kk);   // 逐行取
            half4 a1 = vload4(0, (__local half*)KV + (myc0 + 0) * 64 + kk);
            // 注意：Qt 是 [row][k] 行主序，这里 kk 是固定列，故按行标量取
            av[0] = (float)Qt[(myr0 + 0) * 64 + kk];""",
"""            float av[4], bv[4];
            // Qt 是 [row][k] 行主序，kk 固定 → 按行标量取
            av[0] = (float)Qt[(myr0 + 0) * 64 + kk];""")
s = s.replace("""        (void)a; (void)a1;
        // 暂存 S""", """        // 暂存 S""")
s = s.replace("""            half4 p0 = vload4(0, (__local half*)St + (myr0 + 0) * 64 + kk);
            // 上面这行只是占位；P 的 kk 列需要 4 个行值（标量取）
            float pv[4];""", """            float pv[4];""")
s = s.replace("""            (void)p0;
        }""", """        }""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('cleaned')
