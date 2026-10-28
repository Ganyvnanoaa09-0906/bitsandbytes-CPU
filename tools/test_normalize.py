# -*- coding: utf-8 -*-
"""test_normalize.py — 验证归一化是否真的让 sdist 字节可复现。

步骤（都在干净 git archive 里做）：
  1. 构建两次（固定 SOURCE_DATE_EPOCH）
  2. 各自归一化
  3. 比较两次归一化后的 sha256；同时确认内嵌 DLL 仍是审计值、tar 仍可正常读取
退出码 0 = 可复现；1 = 仍不可复现（并打印差异成员便于继续定位）。
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip()
EXPECT_DLL = sys.argv[1] if len(sys.argv) > 1 else None
WHEEL = sys.argv[2] if len(sys.argv) > 2 else None
EPOCH = 1758000000


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def build_once(tmp, idx):
    tar = os.path.join(tmp, 't%d.tar' % idx)
    subprocess.run(['git', '-C', ROOT, 'archive', '-o', tar, '--format=tar', 'HEAD'], check=True)
    tree = os.path.join(tmp, 'tree%d' % idx)
    os.makedirs(tree)
    subprocess.run(['tar', '-xf', tar, '-C', tree], check=True)
    pkg = os.path.join(tree, 'bitsandbytes')
    if WHEEL:
        with zipfile.ZipFile(WHEEL) as z:
            n = [x for x in z.namelist() if x.endswith('libbitsandbytes_cpu.dll')][0]
            io.open(os.path.join(pkg, 'bitsandbytes', 'libbitsandbytes_cpu.dll'), 'wb').write(z.read(n))
    for d in os.listdir(pkg):
        if d.endswith('.egg-info'):
            shutil.rmtree(os.path.join(pkg, d))
    out = os.path.join(tmp, 'd%d' % idx)
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(EPOCH), PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable, '-m', 'build', '--sdist', '--outdir', out],
                   cwd=pkg, env=env, check=True, capture_output=True)
    f = os.path.join(out, [x for x in os.listdir(out) if x.endswith('.tar.gz')][0])
    norm = os.path.join(tmp, 'n%d.tar.gz' % idx)
    subprocess.run([sys.executable, os.path.join(HERE, 'normalize_sdist.py'), f, norm], check=True,
                   capture_output=True)
    return f, norm


def members(p):
    out = {}
    with tarfile.open(p) as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name] = hashlib.sha256(t.extractfile(m).read()).hexdigest()
    return out


def main():
    tmp = tempfile.mkdtemp(prefix='normtest_')
    try:
        f1, n1 = build_once(tmp, 1)
        f2, n2 = build_once(tmp, 2)
        print('raw   A %d B %s' % (os.path.getsize(f1), sha(f1)[:16]))
        print('raw   B %d B %s' % (os.path.getsize(f2), sha(f2)[:16]))
        print('norm  A %d B %s' % (os.path.getsize(n1), sha(n1)[:16]))
        print('norm  B %d B %s' % (os.path.getsize(n2), sha(n2)[:16]))
        same_raw = sha(f1) == sha(f2)
        same_norm = sha(n1) == sha(n2)
        print('raw  相同: %s' % same_raw)
        print('norm 相同: %s' % same_norm)
        if EXPECT_DLL:
            with tarfile.open(n1) as t:
                dlls = [m for m in t.getmembers() if m.name.endswith('.dll')]
                inner = hashlib.sha256(t.extractfile(dlls[0]).read()).hexdigest()
            print('归一化后内嵌 DLL: %s  %s' % (inner[:16], 'OK' if inner == EXPECT_DLL else 'MISMATCH'))
            if inner != EXPECT_DLL:
                return 1
        if not same_norm:
            a, b = members(n1), members(n2)
            diff = [k for k in a if k in b and a[k] != b[k]]
            print('仍不同成员数: %d %s' % (len(diff), diff[:5]))
            return 1
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
