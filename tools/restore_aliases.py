# -*- coding: utf-8 -*-
"""restore_aliases.py — 从基线（cdd168fd）还原被改写过程抹掉的 import 别名。

背景：`regroup_batch.rewrite_imports` 早期版本会重写**每一条** Import 语句，且只保留
`a.name`、丢掉 `asname`，于是
    import torch.nn as nn            ->  import torch.nn
    import torch.nn.functional as F  ->  import torch.nn.functional
这类别名被抹掉；`_os`/`_sys`（bootstrap 行）只是最先被发现的一例。
本脚本按基线回填：当前缺 asname、而基线上有 asname 的语句，恢复成基线的写法。

实现：用 AST 定位语句 → 在 alias 上补 asname → `ast.unparse` 生成语句文本 → 按缩进替换原行。
只做"补 asname"，不删不改其它内容；改完必须语法可解析。
"""
import ast
import io
import os
import re
import subprocess

ROOT = r'D:\work\bitsandbytes-CPU'
BASE = 'cdd168fd'
MAP_FILE = os.path.join(ROOT, 'docs_cpu', 'SCRIPT_MAP.md')


def old_path_map():
    m = {}
    if not os.path.exists(MAP_FILE):
        return m
    for line in io.open(MAP_FILE, encoding='utf-8'):
        cells = [c.strip().strip('`') for c in line.split('|')]
        if len(cells) >= 4 and cells[1].endswith('.py') and cells[2].endswith('.py'):
            m[cells[2].replace('/', os.sep)] = cells[1]
    return m


def baseline_text(rel):
    r = subprocess.run(['git', '-C', ROOT, 'show', '%s:%s' % (BASE, rel)],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    return r.stdout if r.returncode == 0 else None


def alias_bindings(text):
    """基线里带 asname 的绑定：{('I', module, last) 或 ('F', module, name): asname}"""
    out = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[('I', a.name, a.name.split('.')[-1])] = a.asname
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.asname:
                    out[('F', node.module or '', a.name)] = a.asname
    return out


def fix_one(text, want):
    """返回 (新文本, 修补处数)。"""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if not hasattr(node, 'lineno'):
            continue
        needs = False
        if isinstance(node, ast.Import):
            for a in node.names:
                if not a.asname:
                    asname = want.get(('I', a.name, a.name.split('.')[-1]))
                    if asname and _used(lines, asname):
                        a.asname = asname
                        needs = True
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                if not a.asname:
                    asname = want.get(('F', node.module, a.name))
                    if asname and _used(lines, asname):
                        a.asname = asname
                        needs = True
        if needs:
            indent = lines[node.lineno - 1][:len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
            edits.append((node.lineno, node.end_lineno, indent + ast.unparse(node) + '\n'))
    if not edits:
        return text, 0
    for start, end, newtext in sorted(edits, reverse=True):
        lines[start - 1:end] = [newtext]
    return ''.join(lines), len(edits)


def _used(lines, ident):
    pat = re.compile(r'(?<![\w.])%s(?![\w])' % re.escape(ident))
    return any(pat.search(l) for l in lines)


def main():
    omap = old_path_map()
    changed = skipped = 0
    details = []
    for rel in subprocess.run(['git', '-C', ROOT, 'ls-files', '*.py'], capture_output=True,
                              text=True, encoding='utf-8').stdout.splitlines():
        if rel.startswith('bitsandbytes/'):
            continue          # 发布代码必须与 tag 逐字节一致，归组工具不得改它
        cur_p = os.path.join(ROOT, rel.replace('/', os.sep))
        try:
            cur = io.open(cur_p, encoding='utf-8-sig').read()
        except OSError:
            continue
        base_rel = omap.get(rel.replace('/', os.sep), rel)
        base = baseline_text(base_rel)
        if not base:
            skipped += 1
            continue
        want = alias_bindings(base)
        if not want:
            continue
        try:
            new, n = fix_one(cur, want)
        except SyntaxError as e:
            print('  [SKIP] 解析失败 %s: %s' % (rel, e.msg))
            continue
        if n:
            io.open(cur_p, 'w', encoding='utf-8', newline='').write(new)
            changed += 1
            details.append((rel, n))
    print('回填别名的文件数: %d（无基线跳过 %d）' % (changed, skipped))
    for r, n in details[:12]:
        print('  %-56s %d 处' % (r, n))


if __name__ == '__main__':
    main()
