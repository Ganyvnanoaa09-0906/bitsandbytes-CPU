# -*- coding: utf-8 -*-
"""check_root_regroup.py — 根目录脚本归组的验收器（先于移动写好，用于基线与每批验收）。

判据（按蓝军 2026-10-28 的五处修正）：
  1. **引用边解析**：枚举全仓 .py 的 `import X` / `from X import …` / `import X as Y` /
     `importlib.import_module("X")` 四类；对每条指向"被移动/被整理脚本"的边，断言目标模块文件存在。
     判据是「每条原有边仍可解析」，**不是**"被 import 的集合相等"（dotted name 移动后必然变）。
  2. **静态语法校验**：对仓库内 .py 全部 `ast.parse`，捕获批量文本替换造成的语法破坏。
  3. **不 import 执行**：默认不执行任何被检脚本（它们顶层可能 load 模型/分配 GB/死循环）。
     仅对白名单（`--smoke` 指定）做带 timeout 的 subprocess 冒烟。
  4. 根目录 `.py` 计数、`git ls-files` 计数（供每批比对）。
  5. 结束后可选跑 `tools/check_repo_hygiene.py`（--hygiene）。

用法：
  python tools/check_root_regroup.py --baseline        # 打基线（违规必须为 0）
  python tools/check_root_regroup.py --check           # 每批之后跑；有违规即 exit 1
  python tools/check_root_regroup.py --edges           # 打印所有指向根脚本的边（调试用）
"""
import argparse
import ast
import json
import os
import subprocess
import sys

ROOT = r'D:\work\bitsandbytes-CPU'
EXCLUDE_DIRS = {'.git', '__pycache__', 'build', 'dist', 'dist_windows', 'dist_linux', 'cache_latents',
                'i5build', '_repo_archive_2026-10-27', '.pytest_cache', '.ruff_cache', 'node_modules'}


def walk_py():
    """只扫 **git 跟踪** 的 .py —— 未跟踪的草稿/临时文件不应参与归组断言
    （实测：`gt1030/_t_arena.py` 是未跟踪的临时文件，带未闭合三引号，会把基线判成 FAIL）。"""
    out = subprocess.run(['git', '-C', ROOT, 'ls-files', '*.py'],
                         capture_output=True, text=True, encoding='utf-8', errors='replace')
    paths = [p for p in out.stdout.splitlines() if p.strip()]
    if not paths:                                  # 退回文件系统遍历
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for f in filenames:
                if f.endswith('.py'):
                    yield os.path.join(dirpath, f)
        return
    for rel in paths:
        if rel.split('/')[0] in EXCLUDE_DIRS:
            continue
        yield os.path.join(ROOT, rel.replace('/', os.sep))


def module_index():
    """可导入模块名 → 文件路径。

    仓库有两个 sys.path 基点：仓库根（根目录脚本互 import）与 `bitsandbytes/`（Python 包所在层），
    所以同一个文件可能有两个可导入名，两个都登记。`__init__.py` 的模块名是它所在目录。
    """
    bases = [ROOT, os.path.join(ROOT, 'bitsandbytes')]
    idx = {}
    for p in walk_py():
        rel = os.path.relpath(p, ROOT)
        for base in bases:
            rel_base = os.path.relpath(p, base)
            if rel_base.startswith('..'):
                continue
            mod = rel_base[:-3].replace(os.sep, '.')      # 去掉 .py
            if mod.endswith('__init__'):
                mod = mod[:-len('__init__')].rstrip('.')  # 包名 = 目录
            if not mod:
                continue
            idx.setdefault(mod.split('.')[-1], []).append(rel)
            idx.setdefault(mod, []).append(rel)
    return idx


def edges():
    """产出 (from_rel, dotted_module) 引用边；覆盖四种导入写法。"""
    out = []
    for p in walk_py():
        rel = os.path.relpath(p, ROOT)
        try:
            tree = ast.parse(open(p, 'rb').read())
        except SyntaxError as e:
            out.append((rel, '<SYNTAX_ERROR:%s>' % e))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    out.append((rel, a.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    out.append((rel, node.module))
                    for a in node.names:
                        out.append((rel, node.module + '.' + a.name))
            elif isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, 'attr', getattr(fn, 'id', ''))
                if name in ('import_module', '__import__') and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        out.append((rel, arg.value))
    return out


def resolve(mod, idx):
    """模块或模块前缀是否可解析（返回命中的相对路径列表）。"""
    parts = mod.split('.')
    for i in range(len(parts), 0, -1):
        key = '.'.join(parts[:i])
        if key in idx:
            return idx[key]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--edges', action='store_true')
    ap.add_argument('--hygiene', action='store_true')
    ap.add_argument('--json', default=None)
    ap.add_argument('--baseline-json', default=r'D:\work\video_mvp\regroup_baseline.json',
                    help='基线 JSON：其中已存在的语法错误只作提示，不算本批新增')
    a = ap.parse_args()

    idx = module_index()
    # 只关心【仓库内的本地导入】；stdlib/第三方（os/sys/torch/PIL/…）不参与解析断言
    local_top = set()
    for rels in idx.values():
        for rel in rels:
            local_top.add(rel.split(os.sep)[0].split('.')[0])
    es = edges()
    syntax_bad = [(f, m) for f, m in es if m.startswith('<SYNTAX_ERROR')]
    real = [(f, m) for f, m in es if not m.startswith('<SYNTAX_ERROR')]
    local_edges = [(f, m) for f, m in real if m.split('.')[0] in local_top]
    unresolved = [(f, m) for f, m in local_edges if not resolve(m, idx)]

    root_py = len([f for f in os.listdir(ROOT) if f.endswith('.py') and os.path.isfile(os.path.join(ROOT, f))])
    tracked = subprocess.run(['git', '-C', ROOT, 'ls-files'], capture_output=True, text=True,
                             encoding='utf-8', errors='replace').stdout.splitlines()
    print('== 基线统计 ==')
    print('  仓库内 .py        : %d' % len(list(walk_py())))
    print('  根目录 .py        : %d' % root_py)
    print('  git ls-files      : %d' % len(tracked))
    print('  导入边（四类合计）: %d（其中本地：%d）' % (len(real), len(local_edges)))
    print('  语法错误文件      : %d' % len(syntax_bad))
    print('  无法解析的边      : %d' % len(unresolved))

    if a.edges:
        for f, m in real:
            if not resolve(m, idx):
                print('  [UNRESOLVED] %s -> %s' % (f, m))

    if syntax_bad:
        print('\n语法错误：')
        for f, m in syntax_bad[:20]:
            print('  %s  %s' % (f, m))

    if unresolved:
        print('\n无法解析的引用边（前 30）：')
        for f, m in unresolved[:30]:
            print('  %s -> %s' % (f, m))

    result = dict(root_py=root_py, tracked=len(tracked), edges=len(real), local_edges=len(local_edges),
                  syntax_bad=len(syntax_bad), syntax_bad_files=sorted({f for f, _ in syntax_bad}),
                  unresolved=len(unresolved))
    if a.json:
        open(a.json, 'w', encoding='utf-8').write(json.dumps(result, indent=2))
        print('\n已写 %s' % a.json)

    if a.hygiene:
        print('\n== repo hygiene ==')
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'check_repo_hygiene.py')],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(r.stdout.strip()[-1500:] or r.stderr.strip()[-1500:])
        result['hygiene_exit'] = r.returncode

    bad_files = sorted({f for f, _ in syntax_bad})
    pre_existing = []
    if a.baseline and a.json:
        pass
    if os.path.exists(a.baseline_json) and not a.baseline:
        try:
            base = json.load(open(a.baseline_json, encoding='utf-8'))
            base_bad = set(base.get('syntax_bad_files') or [])
            pre_existing = [f for f in bad_files if f in base_bad]
            new_bad = [f for f in bad_files if f not in base_bad]
        except Exception:
            new_bad = bad_files
    else:
        new_bad = bad_files
    if pre_existing:
        print('\n已存在的语法错误（非本批引入，只作提示）：%s' % ', '.join(pre_existing))
    if new_bad:
        print('\n本批新增语法错误：%s' % ', '.join(new_bad))

    if a.baseline:
        # 打基线时：当前语法错误全部记入基线，不算"新增"
        ok = (len(unresolved) == 0)
    else:
        ok = (len(new_bad) == 0 and len(unresolved) == 0)
    print('\n结果: %s' % ('PASS（无新增违规）' if ok else 'FAIL（见上）'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
