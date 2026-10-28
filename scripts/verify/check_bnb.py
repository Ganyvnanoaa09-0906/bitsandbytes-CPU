# -*- coding: utf-8 -*-
"""check_bnb.py — 搞清楚训练脚本里 `import bitsandbytes` 到底解析到哪个包。

背景：从 D:\\work\\bitsandbytes-CPU 运行时，`import bitsandbytes` 会先撞上
同名的本地目录（命名空间包），可能拿不到 optim 子模块 → 静默退回 torch 的 AdamW。
这会让训练白丢约 2.7% 的速度。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys

print('cwd =', os.getcwd())
print()
print('--- sys.path 前 6 项 ---')
for i, p in enumerate(sys.path[:6]):
    print('  [%d] %s' % (i, p or '(空)'))

print()
try:
    import bitsandbytes as bnb
    print('bitsandbytes.__file__ =', bnb.__file__)
    print('bitsandbytes.__path__ =', list(getattr(bnb, '__path__', [])))
    print('版本 =', getattr(bnb, '__version__', '?'))
    print('有 optim 子模块:', hasattr(bnb, 'optim'))
    if hasattr(bnb, 'optim'):
        print('optim.__file__ =', getattr(bnb.optim, '__file__', '?'))
        A = getattr(bnb.optim, 'AdamW8bit', None)
        print('AdamW8bit =', A)
        if A is not None:
            import inspect
            print('  定义在:', inspect.getsourcefile(A))
            print('  与 torch.optim.AdamW 相同?', A is __import__('torch').optim.AdamW)
except Exception as e:
    import traceback
    traceback.print_exc()
