# -*- coding: utf-8 -*-
"""让 _watch_runb.py 支持环境变量覆盖，做成通用看护（run C/D 都能用）。

    ARW_OUTDIR / ARW_NOTE / ARW_SMPDIR / ARW_LOG / ARW_TAG / ARW_CKPTS(逗号分隔) / ARW_STEP0
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\_watch_runb.py'
s = io.open(p, encoding='utf-8').read()
old = """OUTDIR = os.path.join(F, 'ar_run_igpu40k_b')
SMPDIR = os.path.join(F, 'runb_samples')
NOTE = r'D:\\work\\collab\\shared\\IGPU_RUNB_PROGRESS.md'
VQVAE = r'D:\\work\\cloud_salvage\\ScPeP7\\vqvae_V32.pt'
CKPTS = [32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000]
TRAINLOG = os.path.join(F, 'ar_igpu_runb.log')"""
new = """OUTDIR = os.environ.get('ARW_OUTDIR', os.path.join(F, 'ar_run_igpu40k_b'))
SMPDIR = os.environ.get('ARW_SMPDIR', os.path.join(F, 'runb_samples'))
NOTE = os.environ.get('ARW_NOTE', r'D:\\work\\collab\\shared\\IGPU_RUNB_PROGRESS.md')
VQVAE = r'D:\\work\\cloud_salvage\\ScPeP7\\vqvae_V32.pt'
TAG = os.environ.get('ARW_TAG', 'runB')
_ck = os.environ.get('ARW_CKPTS', '')
CKPTS = [int(x) for x in _ck.split(',')] if _ck else \\
    [32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000]
TRAINLOG = os.environ.get('ARW_LOG', os.path.join(F, 'ar_igpu_runb.log'))"""
assert old in s
s = s.replace(old, new)
s = s.replace("log('watcher start (run B)')", "log('watcher start (%s)' % TAG)")
s = s.replace("""            f.write('# run B 进度（每 2.5k 步采样 + token 指标）\\n\\n'""",
              """            f.write('# %s 进度（每 2.5k 步采样 + token 指标）\\n\\n' % TAG""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('watcher 支持环境变量覆盖')
