"""i5 上做训练前的依赖体检（写成文件传过去，避免嵌套引号）。"""
import importlib.util
import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print('=== 依赖 ===')
for m in ('numpy', 'psutil', 'torch', 'bitsandbytes', 'safetensors'):
    ok = importlib.util.find_spec(m) is not None
    v = ''
    if ok:
        try:
            mod = __import__(m)
            v = getattr(mod, '__version__', '')
        except Exception as e:
            v = 'import 失败: %s' % str(e)[:40]
    print('  %-14s %s %s' % (m, 'OK ' if ok else '缺 ', v))

print('\n=== 磁盘 ===')
for d in ('C:\\', 'D:\\', 'E:\\'):
    if os.path.exists(d):
        try:
            t, u, f = shutil.disk_usage(d)
            print('  %s 总 %.1f GB / 可用 %.1f GB' % (d, t / 2**30, f / 2**30))
        except Exception:
            pass

print('\n=== 关键：训练脚本 import bitsandbytes，它真能用吗 ===')
try:
    import bitsandbytes as bnb
    print('  import 成功:', bnb.__version__, '| 位置:', os.path.dirname(bnb.__file__))
except Exception as e:
    print('  ✗ 不可用:', type(e).__name__, str(e)[:100])
    print('  ⇒ 要么装我们的 CPU fork，要么把训练脚本里那行 import 改成可选')
