# -*- coding: utf-8 -*-
"""repair_after_regroup.py — 分步修复归组留下的两类问题，每步都做语法验证。

步骤（可单独运行）：
  --aliases   从基线 cdd168fd 回填被抹掉的 import 别名（AST 级，安全）
  --own-dir   在 bootstrap 的 `_sys.path.insert(0, _root)` 之后追加"本文件目录"（文本锚点安全）
  --paths     把写死的 `D:/work/bitsandbytes-CPU/<已移动脚本>.py` 改成 _root 相对
每个文件改完立刻 ast.parse；失败则该文件不写盘并报告。绝不触碰 bitsandbytes/。
"""
import argparse
import ast
import io
import os
import subprocess

ROOT = r'D:\work\bitsandbytes-CPU'
BASE = 'cdd168fd'
MAP_FILE = os.path.join(ROOT, 'docs_cpu', 'SCRIPT_MAP.md')
SKIP = 'bitsandbytes/'


def tracked_py():
    return [r for r in subprocess.run(['git', '-C', ROOT, 'ls-files', '*.py'], capture_output=True,
                                      text=True, encoding='utf-8').stdout.splitlines()
            if not r.startswith(SKIP)]


def read(p):
    return io.open(p, encoding='utf-8-sig').read()


def write_checked(rel, text):
    """写盘前验证语法；通过返回 True。"""
    try:
        ast.parse(text)
    except SyntaxError as e:
        print('  [SKIP] %s 改后语法错误: %s' % (rel, e.msg))
        return False
    io.open(os.path.join(ROOT, rel.replace('/', os.sep)), 'w',
            encoding='utf-8', newline='').write(text)
    return True


def old_path_map():
    m = {}
    if os.path.exists(MAP_FILE):
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


def step_aliases():
    omap = old_path_map()
    changed = 0
    for rel in tracked_py():
        p = os.path.join(ROOT, rel.replace('/', os.sep))
        try:
            cur = read(p)
        except OSError:
            continue
        want = alias_bindings(baseline_text(omap.get(rel.replace('/', os.sep), rel)) or '')
        if not want:
            continue
        try:
            tree = ast.parse(cur)
        except SyntaxError:
            continue
        lines = cur.splitlines(keepends=True)
        edits = []
        for node in ast.walk(tree):
            if not hasattr(node, 'lineno'):
                continue
            needs = False
            if isinstance(node, ast.Import):
                for a in node.names:
                    if not a.asname:
                        asname = want.get(('I', a.name, a.name.split('.')[-1]))
                        if asname:
                            a.asname = asname
                            needs = True
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    if not a.asname:
                        asname = want.get(('F', node.module, a.name))
                        if asname:
                            a.asname = asname
                            needs = True
            if needs:
                indent = lines[node.lineno - 1][:len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
                edits.append((node.lineno, node.end_lineno, indent + ast.unparse(node) + '\n'))
        if not edits:
            continue
        for start, end, newtext in sorted(edits, reverse=True):
            lines[start - 1:end] = [newtext]
        if write_checked(rel, ''.join(lines)):
            changed += 1
    print('aliases: 修复 %d 个文件' % changed)


def step_own_dir():
    ANCHOR = '_sys.path.insert(0, _root)'
    BLOCK = ['_here = _os.path.dirname(_os.path.abspath(__file__))',
             'if _here not in _sys.path:',
             '    _sys.path.insert(0, _here)']
    changed = 0
    for rel in tracked_py():
        p = os.path.join(ROOT, rel.replace('/', os.sep))
        try:
            text = read(p)
        except OSError:
            continue
        if '_here = _os.path.dirname' in text or ANCHOR not in text:
            continue
        lines = text.splitlines(keepends=True)
        out, inserted = [], False
        for ln in lines:
            out.append(ln)
            if not inserted and ln.strip() == ANCHOR:
                out.extend(b + '\n' for b in BLOCK)
                inserted = True
        if inserted and write_checked(rel, ''.join(out)):
            changed += 1
    print('own-dir: 修复 %d 个文件' % changed)


def step_paths():
    omap = old_path_map()
    changed = 0
    for rel in tracked_py():
        p = os.path.join(ROOT, rel.replace('/', os.sep))
        try:
            text = read(p)
        except OSError:
            continue
        orig = text
        for new_rel, old_rel in omap.items():   # omap: 新路径 -> 旧（根目录）路径
            for sep in ('\\\\', '\\', '/'):
                bad = 'D:' + sep + 'work' + sep + 'bitsandbytes-CPU' + sep + old_rel.replace(os.sep, sep)
                if bad in text:
                    good = '_os.path.join(_root, %r)' % new_rel
                    text = text.replace(bad, good)
        if text != orig and write_checked(rel, text):
            changed += 1
    print('paths: 修复 %d 个文件' % changed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--aliases', action='store_true')
    ap.add_argument('--own-dir', action='store_true')
    ap.add_argument('--paths', action='store_true')
    a = ap.parse_args()
    if a.aliases:
        step_aliases()
    if a.own_dir:
        step_own_dir()
    if a.paths:
        step_paths()
