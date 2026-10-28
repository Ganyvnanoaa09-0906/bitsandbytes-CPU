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

BOOTSTRAP = ('import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path\n'
             '_root = _os.path.dirname(_os.path.abspath(__file__))\n'
             'while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):\n'
             '    _parent = _os.path.dirname(_root)\n'
             '    if _parent == _root:\n'
             '        break\n'
             '    _root = _parent\n'
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
    """把对 names 的引用改写成 scripts.<theme>.*。

    两个 must（2026-10-28 B6 首跑踩坏 49 个文件得出的）：
      1. **保留缩进**：Import 语句可能位于函数体/try 块内，改写成第 0 列会让整份文件语法错误；
      2. **保留未被移动的名字**：`import a, b` 只移动 a 时不能把 b 丢掉。
    返回 (新源码, 改动处数)。
    """
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits = []          # (起行, 止行, [新行...])
    for node in ast.walk(tree):
        if not hasattr(node, 'lineno'):        # Module 等容器节点没有位置信息
            continue
        line0 = lines[node.lineno - 1]
        indent = re.match(r'[ \t]*', line0).group(0)
        if isinstance(node, ast.Import):
            mov, keep = [], []
            for al in node.names:
                (mov if al.name.split('.')[0] in names else keep).append(al.name)
            new_lines = []
            if keep:
                new_lines.append(indent + 'import ' + ', '.join(keep))
            if mov:
                new_lines.append(indent + 'from scripts.%s import %s' % (theme, ', '.join(mov)))
            if new_lines:
                edits.append((node.lineno, node.end_lineno, [l + '\n' for l in new_lines]))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in names:
                aliases = ', '.join(a.name + ((' as ' + a.asname) if a.asname else '') for a in node.names)
                edits.append((node.lineno, node.end_lineno,
                              [indent + 'from scripts.%s.%s import %s\n' % (theme, node.module, aliases)]))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, 'attr', getattr(fn, 'id', ''))
            if name in ('import_module', '__import__') and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    top = arg.value.split('.')[0]
                    if top in names:
                        edits.append((node.lineno, node.end_lineno,
                                      [line0.replace('"%s"' % arg.value,
                                                     '"scripts.%s.%s"' % (theme, arg.value))]))
    if not edits:
        return src, 0
    edits.sort(key=lambda e: e[0], reverse=True)
    for start, end, new_lines in edits:
        lines[start - 1:end] = new_lines
    return ''.join(lines), len(edits)


def preflight(files, names):
    """批前预检（蓝军 2026-10-28 建议）：把"门禁事后抓"变成"工具事先说清楚"。

    A. 列出所有 importer，按"批内（本批要移动的文件）/ 批外（留在别处的文件）"分组；
       批外 importer 会被本批改写引用，必须显式列出来。
    B. 对每个 importer 检查其目录下是否有同名兄弟文件：有则该引用运行时优先命中兄弟，
       不能改写成 scripts.*（B3b 踩过的条件）。
    返回 (inbatch_importers, external_importers, shadowed) 三个集合。
    """
    inbatch, external, shadowed = set(), set(), set()
    moved_rels = {'%s' % f for f in files}
    for rel in tracked(['*.py']):
        full = os.path.join(ROOT, rel.replace('/', os.sep))
        try:
            text = io.open(full, encoding='utf-8-sig').read()
        except OSError:
            continue
        hit = [n for n in names if re.search(r'\b%s\b' % re.escape(n), text)]
        if not hit:
            continue
        importer_dir = os.path.dirname(full)
        for n in hit:
            if os.path.exists(os.path.join(importer_dir, n + '.py')):
                shadowed.add((rel, n))
            elif rel in moved_rels:
                inbatch.add(rel)
            else:
                external.add(rel)
    return inbatch, external, shadowed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default=None, help='根目录文件前缀，如 bench（匹配 bench_*.py）')
    ap.add_argument('--prefixes', default=None,
                    help='逗号分隔的多个前缀，合并成一批（如 train,eval,analyze）')
    ap.add_argument('--theme', default=None, help='目标子目录名，如 bench')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--all-root-except', default=None,
                    help='移动根目录所有 .py，除了逗号分隔的这些文件名（B6 用）')
    ap.add_argument('--rest-safe', action='store_true',
                    help='把根目录所有"零入边"文件成批移走（B5 收尾）')
    ap.add_argument('--limit', type=int, default=0, help='每批最多移动多少个（0=不限）')
    ap.add_argument('--allow-external-importers', action='store_true',
                    help='允许批外 importer 存在（它们会被本批改写；默认存在即中止）')
    ap.add_argument('--exclude-inbound', action='store_true',
                    help='只移动没有被其他脚本 import 的文件（有入边的库留到 B6 统一处理）')
    a = ap.parse_args()

    if a.all_root_except:
        keep = {x.strip() for x in a.all_root_except.split(',') if x.strip()}
        files = sorted(p for p in tracked(['*.py'])
                       if os.sep not in p and '/' not in p and p not in keep)
        prefix_list = []
    elif a.rest_safe:
        import json
        safe = set(json.load(io.open(r'D:\work\video_mvp\regroup_safeset.json', encoding='utf-8'))['safe'])
        files = [p for p in tracked(['*.py']) if os.sep not in p and '/' not in p
                 and os.path.splitext(p)[0] in safe]
        files = sorted(files)
        if a.limit:
            files = files[:a.limit]
        prefix_list = []
    else:
        prefix_list = ([a.prefix] if a.prefix else []) + ([x.strip() for x in a.prefixes.split(',')] if a.prefixes else [])
        if not prefix_list:
            raise SystemExit('需要 --prefix / --prefixes / --rest-safe')
    if not a.theme:
        raise SystemExit('需要 --theme')
    if prefix_list:
        files = []
        for pfx in prefix_list:
            files += [p for p in tracked(['%s_*.py' % pfx]) if os.sep not in p and '/' not in p]
        files = sorted(set(files))
    if a.exclude_inbound:
        import json
        safe = set(json.load(io.open(r'D:\work\video_mvp\regroup_safeset.json', encoding='utf-8'))['safe'])
        before_n = len(files)
        files = [f for f in files if os.path.splitext(f)[0] in safe]
        print('--exclude-inbound: %d -> %d（跳过有入边的库脚本 %d 个）'
              % (before_n, len(files), before_n - len(files)))
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

    # 批前预检：把跨批依赖与被遮蔽的引用在动手之前说清楚
    inb, ext, shd = preflight(files, names)
    print('  预检A 批内 importer: %d 个文件' % len(inb))
    if ext:
        print('  预检A **批外 importer: %d 个文件**（本批会改写它们的引用）' % len(ext))
        for r in sorted(ext)[:12]:
            print('    - %s' % r)
        if not a.allow_external_importers:
            raise SystemExit('存在批外 importer，默认中止；确认要一并改写请加 --allow-external-importers')
    if shd:
        print('  预检B 被同级同名文件遮蔽（不改写）: %d 处' % len(shd))
        for rel, n in sorted(shd)[:8]:
            print('    - %s 引用 %s（同目录已有同名文件）' % (rel, n))

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
        # 只重写"真的解析到被移动文件"的引用：
        # importer 目录里若已有同名文件，运行时优先命中它（如 gt1030/wan_vae_profile.py），
        # 把它改写成 scripts.<theme>.X 会反而把它变成跨目录依赖。
        importer_dir = os.path.dirname(os.path.abspath(target))
        # 注意：重写发生在 git mv 之后，根目录原文件已不存在 —— 不能再用"根文件存在"作条件。
        # 判据只剩一条：importer 自己目录里是否有同名兄弟文件（有则运行时优先命中它）。
        movable = {n for n in names
                   if not os.path.exists(os.path.join(importer_dir, n + '.py'))}
        try:
            new_text, n_edits = rewrite_imports(src_text, movable, a.theme, target)
        except SyntaxError:
            continue
        if n_edits:
            # 写盘前自检：任何一次改写都不许产出语法错误的文件（B6 首跑就是在这里丢了 49 个文件）
            try:
                import ast
                _ast.parse(new_text)
            except SyntaxError as exc:
                print('  [SKIP] 改写后语法错误，未写盘: %s (%s)' % (target, exc.msg))
                continue
            # 被改写的 importer 可能不在 scripts/ 下（如 gt1030/），它需要仓库根在 sys.path 才能
            # import scripts.<theme>.*；移动集合内的文件已各自插过 bootstrap，这里只补外部文件。
            if 'scripts' + os.sep not in target and '_root = _os.path.dirname' not in new_text:
                try:
                    new_text = insert_bootstrap(new_text)
                except SyntaxError:
                    pass
            io.open(target, 'w', encoding='utf-8', newline='').write(new_text)
            changed += 1
    print('已改写 %d 个文件里的引用（不含新建的 __init__.py）' % changed)
    # 口径分明：新建 / 重命名 / 改写是三类不同的事实
    st = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain'], capture_output=True,
                        text=True, encoding='utf-8', errors='replace').stdout.splitlines()
    renamed = sum(1 for x in st if x[:2].strip() in ('R', 'RM', 'RM'))
    added = sum(1 for x in st if x[:2].strip() == 'A' or x.startswith('??'))
    modified = sum(1 for x in st if x[:2].strip() == 'M')
    print('  git status: 重命名 %d / 新建 %d / 修改 %d' % (renamed, added, modified))

    # 5) 门禁
    print('\n== 归组验收 ==')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'check_root_regroup.py'), '--check'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = r.stdout.strip().splitlines()[-6:]
    print('\n'.join(tail))
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
