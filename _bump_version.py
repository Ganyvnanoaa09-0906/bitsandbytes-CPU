"""Bump the package version so the upload is accepted.

PyPI rejects re-uploading an existing version (0.50.2.dev1 is already there), so
the release needs a new number. Find every place the version is written rather
than assuming a single source -- a wheel whose metadata says dev1 while the
package says dev2 would be its own kind of confusing.

Usage: python _bump_version.py 0.50.2.dev2
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
NEW = sys.argv[1] if len(sys.argv) > 1 else '0.50.2.dev2'
OLD = '0.50.2.dev1'
ROOT = r'D:\work\bnb-4bitopt\bitsandbytes'

hits = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames
                   if d not in ('dist', 'build', '__pycache__', '.git', 'i5build')]
    for fn in filenames:
        if not fn.endswith(('.py', '.cfg', '.toml', '.txt')):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            t = open(fp, encoding='utf-8', errors='replace').read()
        except Exception:
            continue
        if OLD in t:
            hits.append((fp, t.count(OLD)))

if not hits:
    print('  ✗ 没找到 %s' % OLD)
    sys.exit(1)

for fp, n in hits:
    t = open(fp, encoding='utf-8', errors='replace').read()
    t2 = t.replace(OLD, NEW)
    open(fp, 'w', encoding='utf-8', newline='').write(t2)
    print('  + %s  (%d 处)  %s -> %s'
          % (fp.replace(ROOT + '\\', ''), n, OLD, NEW))

# also refresh the copy inside the package if it exists
vp = os.path.join(ROOT, 'bitsandbytes', 'version.py')
if os.path.exists(vp):
    print('  version.py 内容:')
    for line in open(vp, encoding='utf-8').read().splitlines()[:6]:
        print('    ' + line)
