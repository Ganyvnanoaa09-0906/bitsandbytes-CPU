# -*- coding: utf-8 -*-
"""fix_f821.py — 用 ruff F821 的**差分**（当前 vs tag 对应文件）机械修复归组引入的未定义名。

为什么不按名字猜：差分给出"哪个文件的哪个名字是新出现的"，修复只需从 tag 同一文件抄回
绑定它的那条 import（本地模块改成 `from scripts.lib …` 形式）。每个文件写盘前 ast.parse。

用法：
  python tools/fix_f821.py --dry-run
  python tools/fix_f821.py --apply
"""
import argparse
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = r'D:\work\bitsandbytes-CPU'
TAG = 'cdd168fd'
MAP_FILE = os.path.join(ROOT, 'docs_cpu', 'SCRIPT_MAP.md')


def ruff_json(paths):
    r = subprocess.run(['ruff', 'check', '--select', 'F821', '--output-format', 'json'] + paths,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    try:
        data = json.loads(r.stdout or '[]')
    except json.JSONDecodeError:
        return {}
    out = {}
    for item in data:
        fname = item['filename']
        name = re.search(r'Undefined name `([^`]+)`', item['message'])
        if name:
            out.setdefault(fname, set()).add(name.group(1))
    return out


def norm(p, root):
    return os.path.relpath(p, root).replace(os.sep, '/')


def tag_snapshot():
    tmp = tempfile.mkdtemp(prefix='tagf821_')
    tar = os.path.join(tmp, 's.tar')
    subprocess.run(['git', '-C', ROOT, 'archive', '-o', tar, '--format=tar', TAG], check=True)
    subprocess.run(['tar', '-xf', tar, '-C', tmp], check=True)
    return tmp


def script_map():
    """new -> old（相对仓库根）"""
    m = {}
    if os.path.exists(MAP_FILE):
        for line in io.open(MAP_FILE, encoding='utf-8'):
            cells = [c.strip().strip('`') for c in line.split('|')]
            if len(cells) >= 4 and cells[1].endswith('.py') and cells[2].endswith('.py'):
                m[cells[2]] = cells[1]
    return m


def moved_basenames():
    return {os.path.splitext(old)[0] for old in script_map().values()}


def bindings(text):
    """名字 -> tag 里绑定它的 import 语句原文"""
    out = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            src = '\n'.join(lines[node.lineno - 1:node.end_lineno])
            if isinstance(node, ast.Import):
                for a in node.names:
                    key = a.asname or a.name.split('.')[0]
                    out.setdefault(key, src)
            else:
                for a in node.names:
                    key = a.asname or a.name
                    out.setdefault(key, src)
    return out


def transform_import(src, moved):
    """把 tag 里的 import 语句改造成 scripts.lib 形式（若涉及已移动的本地模块）。"""
    try:
        node = ast.parse(src).body[0]
    except SyntaxError:
        return src
    if isinstance(node, ast.Import):
        parts = []
        for a in node.names:
            base = a.name.split('.')[0]
            if base in moved:
                parts.append('from scripts.lib import %s%s' % (
                    base, (' as ' + a.asname) if a.asname else ''))
            else:
                parts.append('import %s%s' % (a.name, (' as ' + a.asname) if a.asname else ''))
        return '\n'.join(parts)
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return src
        base = (node.module or '').split('.')[0]
        names = ', '.join(a.name + ((' as ' + a.asname) if a.asname else '') for a in node.names)
        if base in moved:
            return 'from scripts.lib.%s import %s' % (node.module, names)
        return 'from %s import %s' % (node.module, names)
    return src


def insert_after_imports(text, new_lines):
    """把新 import 行插在最后一个顶层 import 之后（没有则插在 bootstrap 之后）。"""
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, node.end_lineno)
    if not last:
        for i, ln in enumerate(lines, 1):
            if ln.strip() == '_sys.path.insert(0, _root)':
                last = i
                break
    if not last:
        last = 0
    add = [l if l.endswith('\n') else l + '\n' for l in new_lines]
    return ''.join(lines[:last]) + ''.join(add) + ''.join(lines[last:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    tag_dir = tag_snapshot()
    try:
        tag_err = ruff_json(['.'])
        cur_err = ruff_json(['scripts'])
    finally:
        pass
    # ruff 在 tag_dir 下运行
    cwd = os.getcwd()
    os.chdir(tag_dir)
    try:
        tag_err = ruff_json(['.'])
    finally:
        os.chdir(cwd)
        shutil.rmtree(tag_dir, ignore_errors=True)

    smap = script_map()
    moved = moved_basenames()

    def tag_for(cur_rel):
        old = smap.get(cur_rel)
        if old:
            return old
        base = os.path.basename(cur_rel)
        for o in smap.values():
            if os.path.basename(o) == base:
                return o
        return base

    def old_text(old_rel):
        r = subprocess.run(['git', '-C', ROOT, 'show', '%s:%s' % (TAG, old_rel)],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        return r.stdout if r.returncode == 0 else None

    new_total = 0
    plans = []
    for fname, names in cur_err.items():
        rel = norm(fname, ROOT)
        old_rel = tag_for(rel)
        tag_names = set(tag_err.get(old_rel, set()))
        new_names = names - tag_names
        if not new_names:
            continue
        new_total += len(new_names)
        plans.append((rel, old_rel, sorted(new_names)))

    print('当前 F821: %d 文件 / %d 处' % (len(cur_err), sum(len(v) for v in cur_err.values())))
    print('tag  F821: %d 文件 / %d 处' % (len(tag_err), sum(len(v) for v in tag_err.values())))
    print('**真新增**: %d 文件 / %d 处' % (len(plans), new_total))

    fixed_files = fixed_names = 0
    for rel, old_rel, names in plans:
        cur_p = os.path.join(ROOT, rel)
        cur = io.open(cur_p, encoding='utf-8-sig').read()
        old = old_text(old_rel)
        if old is None:
            print('  [SKIP] 没有 tag 对应文件: %s' % rel)
            continue
        bind = bindings(old)
        todo = []
        for n in names:
            src = bind.get(n)
            if not src:
                continue
            todo.append((n, transform_import(src, moved)))
        if not todo:
            print('  [SKIP] tag 里也找不到绑定: %s %s' % (rel, names))
            continue
        # 若当前已有 `from scripts.lib import X`（无 asname），而 todo 是 `... import X as N`，就地补 asname
        lines = cur.splitlines(keepends=True)
        remaining = []
        for n, imp in todo:
            m = re.match(r'from scripts\.lib import ([A-Za-z0-9_]+) as ([A-Za-z0-9_]+)$', imp)
            done = False
            if m:
                mod, alias = m.group(1), m.group(2)
                for i, ln in enumerate(lines):
                    if ln.strip() == 'from scripts.lib import %s' % mod:
                        lines[i] = ln.replace('from scripts.lib import %s' % mod,
                                              'from scripts.lib import %s as %s' % (mod, alias))
                        done = True
                        break
            if not done:
                remaining.append(imp)
        text = ''.join(lines)
        if remaining:
            text = insert_after_imports(text, remaining)
        try:
            ast.parse(text)
        except SyntaxError as e:
            print('  [SKIP] %s 修复后语法错误: %s' % (rel, e.msg))
            continue
        if text != cur:
            fixed_files += 1
            fixed_names += len(todo)
            if a.apply:
                io.open(cur_p, 'w', encoding='utf-8', newline='').write(text)
                print('  修 %-48s +%d 个绑定 %s' % (rel, len(todo), ','.join(n for n, _ in todo)))
    print('\n可修文件 %d / 处 %d（--apply 才写盘）' % (fixed_files, fixed_names))


if __name__ == '__main__':
    main()
