# -*- coding: utf-8 -*-
"""Don't enable XGEMM override by default: KREG>1 makes Xgemm WRONG (rel~1.0)."""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("""    kreg = int(os.environ.get('AR_XGEMM_KREG', '32')) if kreg is None else int(kreg)""",
"""    # ⚠️ 默认关闭：实测 KREG>1 的覆盖参数会让 Xgemm 算错（rel≈1.0），
    # 扫出来的 300~500 GFLOPS 是空转。证据见 _t_override_correct.py。
    kreg = int(os.environ.get('AR_XGEMM_KREG', '0')) if kreg is None else int(kreg)""")
s = s.replace("""                print('[xgemm] override KREG=%s applied' % os.environ.get('AR_XGEMM_KREG', '32'),
                      flush=True)""",
"""                print('[xgemm] override KREG=%s applied'
                      % os.environ.get('AR_XGEMM_KREG', '0'), flush=True)""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('reverted override default -> off')
