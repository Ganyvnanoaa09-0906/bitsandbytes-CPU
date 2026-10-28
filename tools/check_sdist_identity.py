# -*- coding: utf-8 -*-
"""check_sdist_identity.py — 从干净 git archive 构建 sdist，并**断言内嵌 DLL 的身份**。

为什么需要它（2026-10-28 审计会话发现）：
  之前的"打包卫生"验证只检查了形状（"0 个违规产物 / 1 个原生库"），没检查身份。
  而在工作树里 `bitsandbytes/bitsandbytes/libbitsandbytes_cpu.dll` 是**未跟踪**的旧件
  （dev9 的 3be2a7fe…），于是"验证通过"的 sdist 里嵌的是旧内核。
  ⇒ 形状检查通过、身份错误 —— 这正是"检查器只验形状不验身份"。

本脚本强制：
  1. 从 `git archive HEAD` 导出干净树（不含未跟踪文件），在其中构建 sdist
  2. 只从 `--dll-source`（默认取已发布 wheel 里的 DLL）注入内核，并先校验来源 sha256
  3. 断言 sdist 内嵌 DLL 的 sha256 == `--expect-dll`
  4. 断言没有构建产物/字节码混入
  5. 记录 size / sha256 / 成员数 / 内嵌 DLL 到 JSON，供 Gate-2 引用
任何一条不满足 ⇒ 退出码 1。
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = r'D:\work\bitsandbytes-CPU'
AUDITED_DLL_SHA = '472675df952e1270db1df84ca91a296adb00691e29b901a21a51767c977ca855'
WHEEL = r'D:\work\_release_dev10\bitsandbytes_cpu_fork-0.50.2.dev10-py3-none-win_amd64.whl'
FORBIDDEN = ('.pyc', '.pyo', '.obj', '.o', '.a', '.lib', '.exp', '.pdb', '.so', '.dylib')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def export_clean(dest):
    tar = os.path.join(dest, 'tree.tar')
    subprocess.run(['git', '-C', ROOT, 'archive', '-o', tar, '--format=tar', 'HEAD'], check=True)
    dst = os.path.join(dest, 'tree')
    os.makedirs(dst, exist_ok=True)
    subprocess.run(['tar', '-xf', tar, '-C', dst], check=True)
    os.remove(tar)
    return dst


def dll_from_wheel(whl):
    with zipfile.ZipFile(whl) as z:
        names = [n for n in z.namelist() if n.endswith('libbitsandbytes_cpu.dll')]
        if len(names) != 1:
            raise SystemExit('[FATAL] wheel 里 libbitsandbytes_cpu.dll 数量=%d' % len(names))
        return z.read(names[0]), names[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect-dll', default=AUDITED_DLL_SHA)
    ap.add_argument('--dll-source', default=WHEEL,
                    help='wheel（取包内 DLL）或单个 DLL 文件')
    ap.add_argument('--json', default=r'D:\work\collab\shared\sdist_identity_record.json')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix='sdist_identity_')
    tree = export_clean(tmp)
    pkg = os.path.join(tree, 'bitsandbytes')
    if not os.path.isdir(pkg):
        raise SystemExit('[FATAL] git archive 里没有 bitsandbytes/ 目录')

    # 1) 注入审计过的 DLL（未跟踪文件不得参与构建）
    target = os.path.join(pkg, 'bitsandbytes', 'libbitsandbytes_cpu.dll')
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if a.dll_source.lower().endswith('.whl'):
        blob, where = dll_from_wheel(a.dll_source)
        io.open(target, 'wb').write(blob)
        src_desc = '%s::%s' % (os.path.basename(a.dll_source), where)
    else:
        shutil.copy2(a.dll_source, target)
        src_desc = a.dll_source
    got = sha256_file(target)
    print('注入 DLL: %s' % src_desc)
    print('  注入后 sha256 = %s  (%d B)' % (got[:16], os.path.getsize(target)))
    if got != a.expect_dll:
        raise SystemExit('[FATAL] 注入源本身就不是审计值: %s != %s' % (got, a.expect_dll))

    # 2) 清掉可能残留的 egg-info（旧 SOURCES.txt 会把已删文件拉回来）
    for d in os.listdir(pkg):
        if d.endswith('.egg-info'):
            shutil.rmtree(os.path.join(pkg, d))
    # 3) 干净构建
    outdir = os.path.join(tmp, 'dist')
    subprocess.run([sys.executable, '-m', 'build', '--sdist', '--outdir', outdir],
                   cwd=pkg, check=True, capture_output=True)
    sdists = [f for f in os.listdir(outdir) if f.endswith('.tar.gz')]
    if len(sdists) != 1:
        raise SystemExit('[FATAL] 产出的 sdist 数量=%d' % len(sdists))
    sdist = os.path.join(outdir, sdists[0])

    # 4) 断言身份 + 形状
    with tarfile.open(sdist) as t:
        members = [m for m in t.getmembers() if m.isfile()]
        names = [m.name for m in members]
        dlls = [n for n in names if n.endswith('.dll')]
        if len(dlls) != 1:
            raise SystemExit('[FATAL] sdist 里 .dll 数量=%d: %s' % (len(dlls), dlls))
        inner = hashlib.sha256(t.extractfile(dlls[0]).read()).hexdigest()
        bad = [n for n in names if n.endswith(FORBIDDEN) or '__pycache__' in n]
    size = os.path.getsize(sdist)
    ok_identity = (inner == a.expect_dll)
    ok_shape = not bad
    print('sdist: %d B  成员 %d  sha256=%s' % (size, len(names), sha256_file(sdist)[:16]))
    print('  内嵌 DLL: %s' % os.path.basename(dlls[0]))
    print('  内嵌 DLL sha256 = %s  %s' % (inner[:16], 'OK' if ok_identity else 'MISMATCH(expect %s)' % a.expect_dll[:16]))
    print('  构建产物/字节码: %d  %s' % (len(bad), 'OK' if ok_shape else bad[:5]))

    rec = {
        'built_from': 'git archive HEAD',
        'tree_head': subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True).stdout.strip(),
        'dll_source': src_desc,
        'dll_source_sha256': got,
        'expect_dll_sha256': a.expect_dll,
        'embedded_dll_sha256': inner,
        'identity_ok': ok_identity,
        'shape_ok': ok_shape,
        'forbidden_members': bad,
        'sdist_name': sdists[0],
        'sdist_bytes': size,
        'sdist_sha256': sha256_file(sdist),
        'member_count': len(names),
    }
    io.open(a.json, 'w', encoding='utf-8', newline='').write(json.dumps(rec, ensure_ascii=False, indent=2))
    print('记录写入 %s' % a.json)
    if not (ok_identity and ok_shape):
        raise SystemExit('[FATAL] sdist 身份或形状校验失败')
    if a.keep:
        print('[keep] 临时目录: %s' % tmp)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print('结果: PASS（身份 + 形状）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
