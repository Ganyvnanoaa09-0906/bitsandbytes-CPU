"""通用日志尾查看器：自动识别编码（PowerShell 重定向会写 UTF-16，Python 会写 UTF-8）。

用法: python taillog.py <日志路径> [行数]

为什么要有它：用 `ssh host "python -c ..."` 传内联脚本时，PowerShell 的嵌套引号
一整天里坏了六次。以后一律"写文件 → scp → 远程执行这个文件"。
"""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = sys.argv[1] if len(sys.argv) > 1 else 'train.log'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

raw = open(path, 'rb').read()
if raw[:2] == b'\xff\xfe':
    text = raw.decode('utf-16', 'replace')
elif raw[:3] == b'\xef\xbb\xbf':
    text = raw.decode('utf-8-sig', 'replace')
elif raw.count(b'\x00') > len(raw) // 8:
    text = raw.decode('utf-16-le', 'replace')       # 无 BOM 的 UTF-16
else:
    text = raw.decode('utf-8', 'replace')

lines = [ln for ln in text.splitlines() if ln.strip()]
print('--- %s  (%d 行, 显示末 %d 行) ---' % (path, len(lines), n))
for ln in lines[-n:]:
    print(ln[:150])
