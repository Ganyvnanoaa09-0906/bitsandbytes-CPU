# -*- coding: utf-8 -*-
"""收尾脚本再修：日志只取 [v2] 行；M6 用「从 ckpt_20000 续训 100 步」的正式数字；补采样产物。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\_finalize_ar20k.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""def tail(path, n=12):
    try:
        with io.open(path, encoding='utf-8', errors='replace') as f:
            return '\\n'.join(f.read().splitlines()[-n:])
    except Exception as e:
        return str(e)""",
"""def tail(path, n=12):
    \"\"\"只取训练日志行（[v2 ...），避开 train.jsonl 之类的 JSON 尾巴\"\"\"
    try:
        with io.open(path, encoding='utf-8', errors='replace') as f:
            lines = [ln for ln in f.read().splitlines() if ln.startswith('[v2')]
        return '\\n'.join(lines[-n:])
    except Exception as e:
        return str(e)""")

s = s.replace("""        '| M6 续训吞吐 | **1.350 s/step = 1517 tok/s（80 步持续、CPU 挂起下的独占测量）= CPU 的 2.67×** |',""",
"""        '| M6 续训吞吐 | 从 `ckpt_20000` 续训 100 步：**1.383 s/step = 1481 tok/s**（CPU 是 569 tok/s ⇒ **2.60×**）；'
        '独占 80 步持续测 1.350 s/step = 1517 tok/s（2.67×） |',""")

s = s.replace("""        '| `F:\\\\weibo_cos\\\\ar_sample_igpu_12020.png` | iGPU 权重采样（与 CPU 同源可比） |',""",
"""        '| `F:\\\\weibo_cos\\\\ar_sample_igpu_12020.png` | iGPU 权重采样（与 CPU 同源可比） |',
        '| `F:\\\\weibo_cos\\\\ar_run_igpu\\\\ckpt_20100.pt` | **iGPU 从 ckpt_20000 续训 100 步后导出** |',
        '| `F:\\\\weibo_cos\\\\ar_sample_igpu_20100.png` | 上面这个 ckpt 的采样（与 `ar_sample_20000.png` 同源） |',""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('finalize script fixed')
