# -*- coding: utf-8 -*-
"""regroup_batch.py — 把根目录脚本按前缀成批移入 scripts/<theme>/，并重写全部引用边。

设计（对应 2026-10-28 归组计划与蓝军五处修正）：
  * 移动：`git mv`（保留历史），目标 `scripts/<theme>/`
  * 包化：自动建 `scripts/__init__.py` 与 `scripts/<theme>/__init__.py`
  * **被执行时的 sys.path 引导**：给每个被移动文件插入 bootstrap，
    使 `python scripts/<theme>/x.py` 仍能 `import measure_guard` 这类根目录模块
  * **引用边重写**（不用 grep）：AST 定位 `import X` / `from X import` / `import X as Y` /
    `importlib.import_module("X")` 四类，改写成 `from scripts.<theme> import X` /
    `from scripts.<theme>.X import ...` / `"scripts.<theme>.X"`
  * 文档引用（.md/.txt）按同映射做文本替换
  * 跑 `tools/check_root_regroup.py --check` 作为硬门禁

用法：
  python tools/regroup_batch.py --prefix bench --theme bench --dry-run
  python tools/regroup_batch.py --prefix bench --theme bench
"""
import argparse
import ast
import io
import os
import re
import subprocess
import sys

ROOT = r'D:\work\bitsandbytes-CPU'
sys.path.insert(0, os.path.join(ROOT, 'tools'))

BOOTSTRAP = ('import os as _os, sys as _sys  '
             '# 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path\n'
             '_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))\n'
             'if _root not in _sys.path:\n'
             '    _sys.path.insert(0, _root)\n')


def tracked(patterns):
    out = subprocess.run(['git', '-C', ROOT, 'ls-files'] + patterns, capture_output=True,
                         text=True, encoding='utf-8', errors='replace').stdout
    return [p for p in out.splitlines() if p.strip()]


def insert_bootstrap(src):
    """在 shebang / 模块 docstring / __future__ 之后插入 bootstrap。

    注意：仓库里有文件带 UTF-8 BOM（`\\ufeff`），ast.parse 会报
    "invalid non-printable character U+FEFF" —— 先剥掉再解析。
    """
    if src.startswith('\ufeff'):
        src = src[1:]
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith('#!'):
        insert_at = 1
    # PEP 263：encoding 声明必须在第 1–2 行；bootstrap 要插在它之后，否则声明失效（看着也别扭）
    for i in (insert_at, insert_at + 1):
        if i < len(lines) and 'coding' in lines[i] and lines[i].lstrip().startswith('#'):
            insert_at = i + 1
            break
    body = tree.body
    idx = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        insert_at = body[0].end_lineno
        idx = 1
    while idx < len(body) and isinstance(body[idx], ast.ImportFrom) and body[idx].module == '__future__':
        insert_at = body[idx].end_lineno
        idx += 1
    head = ''.join(lines[:insert_at])
    tail = ''.join(lines[insert_at:])
    if head and not head.endswith('\n'):
        head += '\n'
    return head + BOOTSTRAP + tail


def rewrite_imports(src, names, theme, from_rel):
    """把对 names 的引用改写成 scripts.<theme>.*。返回 (新源码, 改动行数)。"""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits = []          # (起行, 止行, 新文本)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split('.')[0]
                if top in names:
                    as_part = (' as %s' % a.asname) if a.asname else ''
                    edits.append((node.lineno, node.end_lineno,
                                  'from scripts.%s import %s%s' % (theme, a.name, as_part)))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in names:
                newmod = 'scripts.%s.%s' % (theme, node.module)
                edits.append((node.lineno, node.end_lineno,
                              'from %s import %s' % (newmod, ', '.join(
                                  a.name + ((' as ' + a.asname) if a.asname else '') for a in node.names))))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, 'attr', getattr(fn, 'id', ''))
            if name in ('import_module', '__import__') and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    top = arg.value.split('.')[0]
                    if top in names:
                        edits.append((node.lineno, node.end_lineno,
                                      lines[node.lineno - 1].replace(
                                          '"%s"' % arg.value,
                                          '"scripts.%s.%s"' % (theme, arg.value))))
    if not edits:
        return src, 0
    for start, end, text in sorted(edits, reverse=True):
        lines[start - 1:end] = [text if text.endswith('\n') else text + '\n']
    return ''.join(lines), len(edits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True, help='根目录文件前缀，如 bench（匹配 bench_*.py）')
    ap.add_argument('--theme', required=True, help='目标子目录名，如 bench')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    files = [p for p in tracked(['%s_*.py' % a.prefix]) if os.sep not in p and '/' not in p]
    names = {f[:-3] for f in files}
    if not files:
        raise SystemExit('没有匹配的根目录文件: %s_*.py' % a.prefix)
    print('待移动 %d 个文件 → scripts/%s/' % (len(files), a.theme))
    print('  ' + ', '.join(sorted(files)[:8]) + (' …' if len(files) > 8 else ''))

    # 找出所有引用这些名字的文件（排除待移动者自身）
    ref_files = set()
    for p in tracked(['*.py', '*.md', '*.txt']):
        full = os.path.join(ROOT, p.replace('/', os.sep))
        try:
            text = io.open(full, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        if any(re.search(r'\b%s\b' % re.escape(n), text) for n in names):
            ref_files.add(p)
    print('含这些名字的文件 %d 个（将按 AST 精确改写，不做无脑替换）' % len(ref_files))

    if a.dry_run:
        print('\n[dry-run] 不做任何修改。')
        return 0

    # 1) 建包
    for d in ('scripts', os.path.join('scripts', a.theme)):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
        init = os.path.join(ROOT, d, '__init__.py')
        if not os.path.exists(init):
            io.open(init, 'w', encoding='utf-8', newline='').write('')

    # 2) git mv
    moved = []
    for f in files:
        src = os.path.join(ROOT, f)
        dst = os.path.join(ROOT, 'scripts', a.theme, f)
        r = subprocess.run(['git', '-C', ROOT, 'mv', f, 'scripts/%s/%s' % (a.theme, f)],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode != 0:
            raise SystemExit('git mv 失败: %s\n%s' % (f, r.stderr))
        moved.append(dst)
    print('\n已 git mv %d 个文件' % len(moved))

    # 3) 插入 bootstrap
    for dst in moved:
        src_text = io.open(dst, encoding='utf-8-sig').read()
        if '_root = _os.path.dirname' in src_text:
            continue
        io.open(dst, 'w', encoding='utf-8', newline='').write(insert_bootstrap(src_text))
    print('已为 %d 个文件插入 sys.path bootstrap' % len(moved))

    # 4) 重写引用（含被移动文件之间的互相引用）
    changed = 0
    for p in sorted(ref_files):
        prefix = 'scripts/%s/' % a.theme
        if p.startswith(prefix):
            full = os.path.join(ROOT, p.replace('/', os.sep))
        elif (os.sep not in p):
            full = os.path.join(ROOT, p) if os.sep in p or '/' in p else os.path.join(ROOT, p)
        else:
            full = os.path.join(ROOT, p.replace('/', os.sep))
        # 被移动文件现在的新路径
        cands = [full, os.path.join(ROOT, 'scripts', a.theme, p)]
        target = next((c for c in cands if os.path.exists(c)), None)
        if not target:
            continue
        if target.endswith(('.md', '.txt')):
            text = io.open(target, encoding='utf-8').read()
            new = text
            for n in sorted(names):
                new = re.sub(r'(?<![\w./])%s\.py\b' % re.escape(n),
                             'scripts/%s/%s.py' % (a.theme, n), new)
            if new != text:
                io.open(target, 'w', encoding='utf-8', newline='').write(new)
                changed += 1
            continue
        src_text = io.open(target, encoding='utf-8-sig').read()
        try:
            new_text, n_edits = rewrite_imports(src_text, names, a.theme, target)
        except SyntaxError:
            continue
        if n_edits:
            io.open(target, 'w', encoding='utf-8', newline='').write(new_text)
            changed += 1
    print('已改写 %d 个文件里的引用' % changed)

    # 5) 门禁
    print('\n== 归组验收 ==')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'check_root_regroup.py'), '--check'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = r.stdout.strip().splitlines()[-6:]
    print('\n'.join(tail))
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
