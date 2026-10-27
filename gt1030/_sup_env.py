# -*- coding: utf-8 -*-
"""让 supervisor 支持全环境变量化（OUTDIR/LOG/NOTE/SUPLOG/BEST/STEPS/LR），
这样 run D（热状态 lr 1e-5 探针）能直接复用同一套守护逻辑。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\_runc_supervisor.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("""OUTDIR = os.path.join(F, 'ar_run_igpu_c')
LOG = os.path.join(F, 'ar_igpu_runc.log')
NOTE = r'D:\\work\\collab\\shared\\IGPU_RUNC_PROGRESS.md'
SUPLOG = os.path.join(F, 'ar_igpu_runc_supervisor.log')
TRAINLOG = os.environ.get('RUNC_LOG', LOG)""",
"""OUTDIR = os.environ.get('RUNC_OUTDIR', os.path.join(F, 'ar_run_igpu_c'))
TRAINLOG = os.environ.get('RUNC_LOG', os.path.join(F, 'ar_igpu_runc.log'))
NOTE = os.environ.get('RUNC_NOTE', r'D:\\work\\collab\\shared\\IGPU_RUNC_PROGRESS.md')
SUPLOG = os.environ.get('RUNC_SUPLOG', os.path.join(F, 'ar_igpu_runc_supervisor.log'))""")
s = s.replace("""CKPT_START = os.path.join(F, 'ar_run_igpu40k', 'ckpt_25000.pt')""",
              """CKPT_START = os.environ.get('RUNC_START', os.path.join(F, 'ar_run_igpu40k', 'ckpt_25000.pt'))""")
s = s.replace("""           '--warmup', '50' if tag == 'rollback' else '200', '--spike', str(SPIKE),""",
              """           '--warmup', '50' if tag == 'rollback' else os.environ.get('RUNC_WARMUP', '200'),
           '--spike', str(SPIKE),""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
import py_compile
py_compile.compile(p, doraise=True)
print('supervisor 已环境变量化，compile ok')
