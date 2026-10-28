# -*- coding: utf-8 -*-
"""check_sdist_identity.py — 从干净 git archive 构建 sdist，断言内嵌 DLL 的**身份 + 出处**。

2026-10-28 审计会话提出的三条，全部落实：

1. **没有跨版本默认值。** 期望 digest 与 DLL 来源必须显式给出。写死上一个版本的
   digest 是最危险的一类"看起来能跑"：下一个版本要么拒绝正确的 DLL，要么诱使人
   把常量改成工作树里的任意值，门禁就此失效。
2. **不绑定本机绝对路径。** 仓库根默认由 `git rev-parse --show-toplevel` 发现，CI 可直接跑。
3. **追溯链四段**：源码 commit → 构建定义（工具链 + -D）→ DLL sha256 → sdist 内嵌 sha256。
   缺任何一段 ⇒ 退出码 2（而不是静默通过）。

用法：
  python tools/check_sdist_identity.py \
      --expect-dll <sha256> --dll-source <wheel|dll> \
      --dll-source-commit <commit> --dll-defines "<-D 列表，可空串>" \
      --dll-toolchain "<工具链描述>" [--root <repo>] [--json <记录>] [--keep]
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

FORBIDDEN = ('.pyc', '.pyo', '.obj', '.o', '.a', '.lib', '.exp', '.pdb', '.so', '.dylib')


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def repo_root(explicit):
    if explicit:
        return os.path.abspath(explicit)
    r = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('[FATAL] 不是 git 仓库且未给 --root: %s' % r.stderr.strip())
    return r.stdout.strip()


def head_commit(root):
    return subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()


def export_clean(root, dest):
    tar = os.path.join(dest, 'tree.tar')
    subprocess.run(['git', '-C', root, 'archive', '-o', tar, '--format=tar', 'HEAD'], check=True)
    tree = os.path.join(dest, 'tree')
    os.makedirs(tree, exist_ok=True)
    subprocess.run(['tar', '-xf', tar, '-C', tree], check=True)
    os.remove(tar)
    return tree


def dll_blob(source):
    if source.lower().endswith('.whl'):
        with zipfile.ZipFile(source) as z:
            names = [n for n in z.namelist() if n.endswith('libbitsandbytes_cpu.dll')]
            if len(names) != 1:
                raise SystemExit('[FATAL] wheel 里 DLL 数量=%d' % len(names))
            return z.read(names[0]), '%s::%s' % (os.path.basename(source), names[0])
    if not os.path.isfile(source):
        raise SystemExit('[FATAL] DLL 来源不存在: %s' % source)
    return io.open(source, 'rb').read(), os.path.abspath(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect-dll', required=True,
                    help='本次发布审计过的 DLL sha256（显式给出，无默认值）')
    ap.add_argument('--dll-source', required=True, help='wheel 或单个 DLL 文件')
    ap.add_argument('--dll-source-commit', default=None, help='构建该 DLL 的源码 commit')
    ap.add_argument('--dll-defines', default=None, help='构建定义（-D 列表），无则传空串')
    ap.add_argument('--dll-toolchain', default=None, help='工具链描述，例如 "zig 0.14.1 windows-gnu"')
    ap.add_argument('--root', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    root = repo_root(a.root)
    tmp = tempfile.mkdtemp(prefix='sdist_identity_')
    try:
        tree = export_clean(root, tmp)
        pkg = os.path.join(tree, 'bitsandbytes')
        if not os.path.isdir(pkg):
            raise SystemExit('[FATAL] git archive 里没有 bitsandbytes/')

        target = os.path.join(pkg, 'bitsandbytes', 'libbitsandbytes_cpu.dll')
        os.makedirs(os.path.dirname(target), exist_ok=True)
        blob, src_desc = dll_blob(a.dll_source)
        io.open(target, 'wb').write(blob)
        got = hashlib.sha256(blob).hexdigest()
        print('注入 DLL: %s' % src_desc)
        print('  来源 sha256 = %s  (%d B)' % (got[:16], len(blob)))
        if got != a.expect_dll:
            raise SystemExit('[FATAL] 注入源本身就不是审计值: %s != %s' % (got, a.expect_dll))

        for d in os.listdir(pkg):
            if d.endswith('.egg-info'):
                shutil.rmtree(os.path.join(pkg, d))

        outdir = os.path.join(tmp, 'dist')
        subprocess.run([sys.executable, '-m', 'build', '--sdist', '--outdir', outdir],
                       cwd=pkg, check=True, capture_output=True)
        names = [f for f in os.listdir(outdir) if f.endswith('.tar.gz')]
        if len(names) != 1:
            raise SystemExit('[FATAL] sdist 数量=%d' % len(names))
        sdist = os.path.join(outdir, names[0])

        with tarfile.open(sdist) as t:
            files = [m for m in t.getmembers() if m.isfile()]
            mnames = [m.name for m in files]
            dlls = [n for n in mnames if n.endswith('.dll')]
            if len(dlls) != 1:
                raise SystemExit('[FATAL] sdist 里 .dll 数量=%d: %s' % (len(dlls), dlls))
            embedded = hashlib.sha256(t.extractfile(dlls[0]).read()).hexdigest()
            bad = [n for n in mnames if n.endswith(FORBIDDEN) or '__pycache__' in n]

        identity_ok = embedded == a.expect_dll
        shape_ok = not bad
        provenance_complete = bool(a.dll_source_commit) and a.dll_defines is not None \
            and bool(a.dll_toolchain)

        print('sdist: %d B  成员 %d  sha256=%s' % (os.path.getsize(sdist), len(mnames),
                                                   sha_file(sdist)[:16]))
        print('  内嵌 DLL sha256 = %s  %s' % (embedded[:16], 'OK' if identity_ok else 'MISMATCH'))
        print('  构建产物/字节码违规 = %d  %s' % (len(bad), 'OK' if shape_ok else bad[:5]))
        print('  出处完整 = %s (commit=%s toolchain=%s)' %
              (provenance_complete, (a.dll_source_commit or '?')[:12], a.dll_toolchain or '?'))

        rec = {
            'tree_head': head_commit(root),
            'built_from': 'git archive HEAD',
            'dll_build': {
                'source_commit': a.dll_source_commit,
                'defines': a.dll_defines,
                'toolchain': a.dll_toolchain,
            },
            'provenance_complete': provenance_complete,
            'dll_source': src_desc,
            'dll_source_sha256': got,
            'expect_dll_sha256': a.expect_dll,
            'embedded_dll_sha256': embedded,
            'identity_ok': identity_ok,
            'shape_ok': shape_ok,
            'forbidden_members': bad,
            'sdist_name': names[0],
            'sdist_bytes': os.path.getsize(sdist),
            'sdist_sha256': sha_file(sdist),
            'member_count': len(mnames),
        }
        if a.json:
            io.open(a.json, 'w', encoding='utf-8', newline='').write(
                json.dumps(rec, ensure_ascii=False, indent=2))
            print('记录: %s' % a.json)

        if not (identity_ok and shape_ok):
            print('结果: FAIL（身份或形状）')
            return 1
        if not provenance_complete:
            print('结果: FAIL（缺构建出处：--dll-source-commit / --dll-defines / --dll-toolchain）')
            return 2
        print('结果: PASS（身份 + 形状 + 出处）')
        return 0
    finally:
        if a.keep:
            print('[keep] %s' % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
