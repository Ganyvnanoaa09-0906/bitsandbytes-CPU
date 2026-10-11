"""看 i5 上训练到底跑起来没有（写文件传过去，避免引号）。"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import psutil

print('=== python 进程 ===')
found = False
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
    try:
        if 'python' in (p.info['name'] or '').lower():
            cl = ' '.join(p.info['cmdline'] or [])[:110]
            print('  PID %-6d %6.0f MB  %s' % (
                p.info['pid'], p.info['memory_info'].rss / 2**20, cl))
            found = True
    except Exception:
        pass
if not found:
    print('  （没有 python 在跑）')

print('\n=== D:\\work_ar 目录 ===')
for root, dirs, files in os.walk(r'D:\work_ar'):
    depth = root.count(os.sep) - r'D:\work_ar'.count(os.sep)
    if depth > 1:
        continue
    for f in files:
        fp = os.path.join(root, f)
        print('  %-46s %10.1f KB' % (os.path.relpath(fp, r'D:\work_ar'),
                                     os.path.getsize(fp) / 1024))
    if depth == 0:
        print('  [dirs]', dirs)
