# -*- coding: utf-8 -*-
"""check_root_regroup.py — 根目录脚本归组的验收器（v2，2026-10-28 重写）。

v1 的致命缺陷（蓝军证伪测试发现）：`module_index()` 无条件按 **basename** 登记
（`idx.setdefault(mod.split('.')[-1], …)`），于是任何文件无论被移到哪个目录，
`import <name>` 都能"解析成功" —— 门禁因此永远 PASS（他把 109 条断边移走后仍 PASS）。

v2 判据：
  1. **只登记"从该 base 真的可导入"的名字**：
     - `base/foo.py` → `foo`
     - `base/pkg/__init__.py` → `pkg`（要求 base/pkg/__init__.py 存在）
     - `base/pkg/mod.py` → `pkg.mod`（要求 base/pkg/__init__.py 存在）
     **删除 basename 兜底**。
  2. 两个 sys.path 基点：仓库根、`bitsandbytes/`（Python 包所在层）。
  3. **相对导入**（`from .x import y`，`node.level>0`）按 importer 所在包解析；
     importer 不在包内则显式计为 unresolved，不静默当绝对导入。
  4. `--selftest`：S1 当前树应 PASS；S2 在临时 clone 里 `git mv` 一个被 import 的模块且不改引用，
     必须 unresolved>0 / exit≠0 —— S2 不过则本门禁不可用于归组。
  5. 语法错误只算"基线之后新增"；`gt1030/_t_arena.py` 这类既有问题只提示。

用法：
  python tools/check_root_regroup.py --baseline
  python tools/check_root_regroup.py --check
  python tools/check_root_regroup.py --selftest
  python tools/check_root_regroup.py --check --root D:\tmp\clone
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse
import ast
import json
import os
import subprocess
import sys

ROOT = r'D:\work\bitsandbytes-CPU'
PKG_BASE = 'bitsandbytes'          # 第二个 sys.path 基点（Python 包所在层）
EXCLUDE_DIRS = {'.git', '__pycache__', 'build', 'dist', 'dist_windows', 'dist_linux', 'cache_latents',
                'i5build', '_repo_archive_2026-10-27', '.pytest_cache', '.ruff_cache', 'node_modules'}


def tracked_py(root):
    out = subprocess.run(['git', '-C', root, 'ls-files', '*.py'], capture_output=True, text=True,
                         encoding='utf-8', errors='replace').stdout
    for rel in out.splitlines():
        rel = rel.strip()
        if not rel or rel.split('/')[0] in EXCLUDE_DIRS:
            continue
        yield rel.replace('/', os.sep)


def has_init(base, dirs):
    """dirs 的每一级（1..n）都必须有 __init__.py 才算包。"""
    for i in range(1, len(dirs) + 1):
        if not os.path.exists(os.path.join(base, *dirs[:i], '__init__.py')):
            return False
    return True


def build_index(root, base):
    """base 下"真的可导入"的名字 → 相对 root 的路径列表。**没有 basename 兜底。**"""
    idx = {}
    if not os.path.isdir(base):
        return idx
    base_abs = os.path.abspath(base)
    for rel in tracked_py(root):
        abs_p = os.path.abspath(os.path.join(root, rel))
        if not abs_p.startswith(base_abs + os.sep):
            continue
        rel_base = os.path.relpath(abs_p, base_abs)
        parts = rel_base[:-3].split(os.sep)          # 去 .py
        if parts[-1] == '__init__':
            dirs = parts[:-1]
            if dirs and has_init(base_abs, dirs):
                idx.setdefault('.'.join(dirs), []).append(rel)
            continue
        if len(parts) > 1:
            if not has_init(base_abs, parts[:-1]):
                continue                             # 不是包 ⇒ 不可 `import`
            idx.setdefault('.'.join(parts), []).append(rel)
        else:
            idx.setdefault(parts[0], []).append(rel)  # base/foo.py
    return idx


_CACHE = {}
_IDX = {}


def idx_for(root, base):
    """按 base 构建（并缓存）可导入名索引。base 可以是仓库根、包层，或某个脚本目录。"""
    key = (root, os.path.abspath(base))
    if key not in _IDX:
        _IDX[key] = build_index(root, base)
    return _IDX[key]


def indexes(root):
    if root not in _CACHE:
        _CACHE[root] = (idx_for(root, root), idx_for(root, os.path.join(root, PKG_BASE)))
    return _CACHE[root]


def local_tops(root):
    ir, ip = indexes(root)
    return set(k.split('.')[0] for k in ir) | set(k.split('.')[0] for k in ip)


def package_of(rel):
    dirs = rel.split(os.sep)[:-1]
    return '.'.join(dirs)


def edges(root):
    """(from_rel, module, level, lineno)"""
    out = []
    for rel in tracked_py(root):
        p = os.path.join(root, rel)
        try:
            tree = ast.parse(open(p, 'rb').read())
        except SyntaxError as e:
            out.append((rel, '<SYNTAX_ERROR:%s>' % e, 0, 0, False))
            continue
        raw = open(p, 'rb').read().split(b'\n')
        marker_line = 0
        for idx, line in enumerate(raw, start=1):
            if b'import os as _os' in line and b'sys as _sys' in line:
                marker_line = idx
                break
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    instr = bool(marker_line and node.lineno == marker_line
                                 and a.name in ('os', 'sys')
                                 and (a.asname or '') in ('_os', '_sys'))
                    out.append((rel, a.name, 0, node.lineno, instr))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    out.append((rel, node.module or '', node.level, node.lineno, False))
                elif node.module:
                    out.append((rel, node.module, 0, node.lineno, False))
            elif isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, 'attr', getattr(fn, 'id', ''))
                if name in ('import_module', '__import__') and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                            and not arg.value.startswith('.'):
                        out.append((rel, arg.value, 0, node.lineno, False))
    return out


def resolve(root, rel, mod, level):
    """绝对导入：按 importer 的运行时 sys.path 解析 —— [脚本所在目录, 仓库根, 包层]；
    相对导入：按 importer 的包解析（根目录文件的相对导入必然 ImportError，由 analyse 单列 exempt）。"""
    ir, ip = indexes(root)
    importer_dir = os.path.dirname(os.path.abspath(os.path.join(root, rel)))
    if level:
        # 相对导入必须按 **importer 所在包** 解析，而"包名"取决于 importer 是从哪个 base 看的：
        #   base=ROOT              → bitsandbytes/bitsandbytes/__init__.py 的包名是 bitsandbytes.bitsandbytes
        #   base=ROOT/bitsandbytes → 同一个文件的包名是 bitsandbytes
        # 两个 base 都试；任一命中即视为可解析。
        abs_p = os.path.abspath(os.path.join(root, rel))
        tried = []
        for base in (root, os.path.join(root, PKG_BASE)):
            base_abs = os.path.abspath(base)
            if not abs_p.startswith(base_abs + os.sep):
                continue
            relb = os.path.relpath(abs_p, base_abs)
            parts = relb.split(os.sep)[:-1]           # 目录链 = 包
            if level > 1:
                if len(parts) < level - 1:
                    continue
                parts = parts[:len(parts) - (level - 1)]
            if not parts:
                continue
            cand = '.'.join(parts + ([mod] if mod else []))
            tried.append(cand)
            if cand in ir or cand in ip:
                return True, 'relative:' + cand
        return False, 'relative_unresolved:' + '|'.join(tried)
    # 3) 绝对导入：依次试 [importer 目录, 仓库根, 包层]
    seen = []
    for base in (importer_dir, root, os.path.join(root, PKG_BASE)):
        base_abs = os.path.abspath(base)
        if base_abs in seen:
            continue
        seen.append(base_abs)
        bidx = idx_for(root, base_abs)
        if mod in bidx:
            return True, 'exact@%s' % os.path.relpath(base_abs, root)
        parts = mod.split('.')
        for i in range(len(parts) - 1, 0, -1):
            cand = '.'.join(parts[:i])
            if cand in bidx:
                return True, 'prefix:%s@%s' % (cand, os.path.relpath(base_abs, root))
    return False, 'not_found'


def analyse(root, baseline_json=None):
    ir, ip = indexes(root)
    tops = local_tops(root)
    es = edges(root)
    syntax_bad = sorted({f for f, m, _l, _n, _i in es if m.startswith('<SYNTAX_ERROR')})
    all_real = [(f, m, l, n) for f, m, l, n, _i in es if not m.startswith('<SYNTAX_ERROR')]
    instrumentation = [(f, m, l, n) for f, m, l, n, i in es if i]
    real = [(f, m, l, n) for f, m, l, n, i in es if not i and not m.startswith('<SYNTAX_ERROR')]
    # "本地"判定：顶层名命中【当前可导入名】或【仓库里任何 .py 的文件名 stem】。
    # 后者是必要的：模块被移到不可导入的位置后，它的 stem 仍存在于仓库，
    # 引用它的边必须继续被检查（否则 S2 那种断边会被整体跳过）。
    stems = {os.path.basename(rel)[:-3] for rel in tracked_py(root)}
    local = [(f, m, l, n) for f, m, l, n in real
             if l > 0 or (m and (m.split('.')[0] in tops or m.split('.')[0] in stems))]
    unresolved = []
    exempt = []
    for f, m, l, n in local:
        ok, why = resolve(root, f, m, l)
        if ok:
            continue
        if l > 0 and not package_of(f):
            # 根目录文件（非包内）的 `from . import x` 在 Python 语义下必定 ImportError，
            # 这类代码必然自带 except ImportError 的绝对导入 fallback（实测 bitsandbytes_cpu_cli.py:1086）。
            # 单列 exempt，不计失败，但要报出来供人核。
            exempt.append((f, m, l, n, why))
        else:
            unresolved.append((f, m, l, n, why))
    tracked = subprocess.run(['git', '-C', root, 'ls-files'], capture_output=True, text=True,
                             encoding='utf-8', errors='replace').stdout.splitlines()
    # root_py 也走 git 口径：文件系统统计会把未跟踪的 .py 算进来（会与 tracked 混用两个数据源）
    root_py = len([t for t in tracked if t.endswith('.py') and '/' not in t])
    fs_root_py = len([f for f in os.listdir(root) if f.endswith('.py')
                      and os.path.isfile(os.path.join(root, f))])
    fail_closed = (len(tracked) == 0 and fs_root_py > 0)
    return dict(root_py=root_py, tracked=len(tracked), py_files=sum(1 for _ in tracked_py(root)),
                fail_closed=fail_closed, fs_root_py=fs_root_py,
                edges=len(real), instrumentation_edges=len(instrumentation),
                local_edges=len(local), syntax_bad_files=syntax_bad,
                unresolved=unresolved, exempt=exempt, idx_root=len(ir), idx_pkg=len(ip))


def git_head(root):
    try:
        return subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True, encoding='utf-8', errors='replace').stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def report(st, baseline_json=None, root=None):
    # 规则 10（蓝军）：任何"扫描某个根"的工具必须打印它实际扫了哪里 ——
    # --root 用错时输出格式与 PASS 完全一样，看着像成功。
    if root:
        print('== 扫描根 ==')
        print('  root   = %s   (来源：%s)' % (os.path.abspath(root),
              '--root 显式传入' if os.path.abspath(root) != os.path.abspath(ROOT) else '--root 默认值'))
        print('  commit = %s' % git_head(root))
    print('== 统计 ==')
    print('  仓库内 .py      : %d' % st['py_files'])
    print('  根目录 .py      : %d' % st['root_py'])
    print('  git ls-files    : %d' % st['tracked'])
    print('  导入边          : %d（本地 %d）  ［已剔除仪器边 %d］'
          % (st['edges'], st['local_edges'], st.get('instrumentation_edges', 0)))
    print('  可导入名        : root=%d / pkg=%d' % (st['idx_root'], st['idx_pkg']))
    print('  语法错误文件    : %d' % len(st['syntax_bad_files']))
    if st.get('fail_closed'):
        print('  [FATAL] 枚举到 0 个被跟踪文件，但根目录有 %d 个 .py —— 当前目录不是 git 工作树？'
              % st.get('fs_root_py', 0))
        print('          本门禁无法枚举导入边，**拒绝报 PASS**（fail closed）。')
        return False
    print('  **无法解析的边**: %d' % len(st['unresolved']))
    print('  根目录相对导入（exempt）: %d' % len(st.get('exempt') or []))
    for f, m, l, n, why in (st.get('exempt') or [])[:10]:
        print('    [EXEMPT] %s:%d  .%s' % (f, n, m))
    for f, m, l, n, why in st['unresolved'][:30]:
        print('    [UNRESOLVED] %s:%d  %s  (%s)' % (f, n, '.' * l + m, why))
    if len(st['unresolved']) > 30:
        print('    … 其余 %d 条省略' % (len(st['unresolved']) - 30))
    new_bad = []
    if baseline_json and os.path.exists(baseline_json):
        base = json.load(open(baseline_json, encoding='utf-8'))
        base_bad = set(base.get('syntax_bad_files') or [])
        new_bad = [f for f in st['syntax_bad_files'] if f not in base_bad]
        if base_bad:
            print('  既有语法错误（基线内，仅提示）: %s' % ', '.join(sorted(base_bad)))
    else:
        new_bad = st['syntax_bad_files']
    if new_bad:
        print('  新增语法错误: %s' % ', '.join(new_bad))
    return (len(st['unresolved']) == 0 and not new_bad)


def run_selftest(root, baseline_json):
    print('== S1：当前树应 PASS ==')
    st = analyse(root, baseline_json)
    ok1 = report(st, baseline_json, root=root)
    print('  S1: %s' % ('PASS' if ok1 else 'FAIL'))
    if not ok1:
        return False

    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='rgselftest_')
    dest = os.path.join(tmp, 'clone')
    try:
        print('\n== S2：移动一个被 import 的模块、不改引用 ⇒ 必须 FAIL ==')
        subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', root, dest], check=True)
        target = 'small_image_model_v2.py'
        if not os.path.exists(os.path.join(dest, target)):
            cands = sorted(f for f in os.listdir(dest) if f.endswith('.py'))
            if not cands:
                print('  S2: SKIP（副本无 .py）')
                return True
            target = cands[0]
        os.makedirs(os.path.join(dest, 'scripts_moved'), exist_ok=True)
        subprocess.run(['git', '-C', dest, 'mv', target, 'scripts_moved/' + target], check=True)
        _CACHE.clear()
        _IDX.clear()
        st2 = analyse(dest, baseline_json)
        print('  unresolved = %d（移动 %s）' % (len(st2['unresolved']), target))
        ok2 = len(st2['unresolved']) > 0
        print('  S2: %s' % ('PASS（抓住断边）' if ok2 else 'FAIL（门禁对断边无感）'))

        print('\n== S3：无 .git 的树（git archive 导出）⇒ 必须 FATAL ==')
        bare = os.path.join(tmp, 'nogit')
        os.makedirs(bare, exist_ok=True)
        tar = os.path.join(tmp, 'src.tar')
        subprocess.run(['git', '-C', root, 'archive', '-o', tar, '--format=tar', 'HEAD'], check=True)
        subprocess.run(['tar', '-xf', tar, '-C', bare], check=True)
        _CACHE.clear()
        _IDX.clear()
        st3 = analyse(bare, baseline_json)
        ok3 = bool(st3.get('fail_closed'))
        print('  tracked=%d / 根目录 .py=%d / fail_closed=%s'
              % (st3['tracked'], st3.get('fs_root_py'), st3.get('fail_closed')))
        print('  S3: %s' % ('PASS（fail closed）' if ok3 else 'FAIL（无 git 时静默 PASS）'))
        return ok1 and ok2 and ok3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--json', default=None)
    ap.add_argument('--baseline-json', default=r'D:\work\video_mvp\regroup_baseline.json')
    a = ap.parse_args()

    # `--baseline` 必须真的把基线写回去：它是"刷新记录"的唯一动作。此前它只跳过比较、
    # 不落盘，于是 --baseline 之后 JSON 里的 commit 仍是旧值（结构性原因：写入分支只看
    # a.json）。这里在解析后立即补默认值，保证写入分支能走到。
    if a.baseline and not a.json:
        a.json = a.baseline_json

    if a.selftest:
        ok = run_selftest(a.root, a.baseline_json)
        print('\n自测结果: %s' % ('PASS' if ok else 'FAIL'))
        return 0 if ok else 1

    st = analyse(a.root, a.baseline_json)
    ok = report(st, None if a.baseline else a.baseline_json, root=a.root)
    if a.json:
        payload = dict(st, commit=git_head(a.root), root=os.path.abspath(a.root))
        try:
            payload['tree'] = subprocess.run(['git', '-C', a.root, 'rev-parse', 'HEAD^{tree}'],
                                             capture_output=True, text=True, encoding='utf-8',
                                             errors='replace').stdout.strip()
        except Exception:
            payload['tree'] = None

        # 口径必须写进记录，否则后来人无法复现同一个数字（蓝军 2026-10-28）
        payload['edge_counting'] = (
            'Import: 每个 alias 计 1 条；ImportFrom: 每个语句计 1 条（按 module，不按 alias）；'
            'importlib.import_module("x"): 计 1 条；归组 bootstrap 的 os/sys alias 归入 '
            'instrumentation_edges，不进入 edges；edges 为剔除仪器边后的 effective 值')
        payload['root_py_source'] = 'git ls-files 中不含目录分隔符的 .py（与 tracked 同源）'
        payload['bootstrap_marker'] = (
            '行内同时含 "import os as _os" 与 "sys as _sys" 的行号；'
            '按行内容匹配而非整行相等，因此容忍尾注释与插入位置')
        payload['unresolved'] = [list(x) for x in st['unresolved']]
        open(a.json, 'w', encoding='utf-8').write(json.dumps(payload, ensure_ascii=False, indent=2))
        print('\n已写 %s' % a.json)
    if a.baseline:
        ok = True
    print('\n结果: %s' % ('PASS（无新增违规）' if ok else 'FAIL（见上）'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
