# -*- coding: utf-8 -*-
"""fix_moved_script_paths.py — 修两类"移动后才暴露"的问题。

1) **同目录裸导入**：库脚本移到 scripts/lib 后成了包成员，`import forge_gemm` 这类同目录裸导入
   只在"直接跑文件"（sys.path[0]=scripts/lib）时成立，用 `import scripts.lib.forge_patch` 就断。
   修法：给 bootstrap 补一行"把本文件所在目录也加入 sys.path"——两种运行方式都对。
2) **硬编码绝对路径**：脚本里写死 `D:\\work\\bitsandbytes-CPU\\anime_adiff.py`，而该文件已移走。
   修法：把"仓库根 + 已移动文件名"的字符串改成 `_root` 相对拼接。

只改 scripts/ 与 gt1030/ 等非发布目录；`bitsandbytes/` 一律跳过。
"""
import io
import os
import re
import subprocess

ROOT = r'D:\work\bitsandbytes-CPU'
SKIP_PREFIX = ('bitsandbytes/',)
MARKER = '_root = _os.path.dirname(_os.path.abspath(__file__))'
HERE_BLOCK = ('_here = _os.path.dirname(_os.path.abspath(__file__))\n'
              'if _here not in _sys.path:\n'
              '    _sys.path.insert(0, _here)\n')


def moved_names():
    """已移动脚本的 basename -> 新相对路径（来自 SCRIPT_MAP）。"""
    m = {}
    p = os.path.join(ROOT, 'docs_cpu', 'SCRIPT_MAP.md')
    for line in io.open(p, encoding='utf-8'):
        cells = [c.strip().strip('`') for c in line.split('|')]
        if len(cells) >= 4 and cells[1].endswith('.py') and cells[2].endswith('.py'):
            m[cells[1]] = cells[2]
    return m


def main():
    moved = moved_names()
    tracked = subprocess.run(['git', '-C', ROOT, 'ls-files', '*.py'], capture_output=True,
                             text=True, encoding='utf-8').stdout.splitlines()
    n_here = n_path = 0
    for rel in tracked:
        if rel.startswith(SKIP_PREFIX):
            continue
        p = os.path.join(ROOT, rel.replace('/', os.sep))
        try:
            text = io.open(p, encoding='utf-8-sig').read()
        except OSError:
            continue
        orig = text

        # 1) 补 own-dir 到 sys.path（仅对已有 bootstrap 的文件）
        if MARKER in text and '_here = _os.path.dirname' not in text:
            text = text.replace(MARKER, HERE_BLOCK + MARKER, 1)
            if text != orig:
                n_here += 1
                orig = text

        # 2) 硬编码绝对路径 -> _root 相对
        for base, new_rel in moved.items():
            for sep in ('\\\\', '\\', '/'):
                bad = 'D:' + sep + 'work' + sep + 'bitsandbytes-CPU' + sep + base
                if bad in text:
                    good = '_os.path.join(_root, %r)' % new_rel.replace('/', os.sep)
                    text = text.replace(bad, good)
                    n_path += 1

        if text != orig:
            io.open(p, 'w', encoding='utf-8', newline='').write(text)

    print('补 own-dir sys.path 的文件数: %d' % n_here)
    print('改写硬编码绝对路径处数: %d' % n_path)


if __name__ == '__main__':
    main()
