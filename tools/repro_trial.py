# -*- coding: utf-8 -*-
"""repro_trial.py — dev11 项：SOURCE_DATE_EPOCH 能否让 sdist 字节可复现。

做法：在干净 `git archive` 树里注入审计 DLL（与发布一致），用固定 SOURCE_DATE_EPOCH
构建两次 sdist，比较 sha256；再与"不设该变量"的两次构建比较。
输出 JSON 记录，供 dev11 决定"发布前是否允许重建"。

注意：wheel 历史上不可字节复现（zip 成员 mtime 不同）；本试验先看 sdist。
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = r'D:\work\bitsandbytes-CPU'
WHEEL = r'D:\work\_release_dev10\bitsandbytes_cpu_fork-0.50.2.dev10-py3-none-win_amd64.whl'
EPOCH = '1758000000'   # 2025-09-16，固定值本身不重要，重要的是两次相同


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def export(dest):
    tar = os.path.join(dest, 't.tar')
    subprocess.run(['git', '-C', ROOT, 'archive', '-o', tar, '--format=tar', 'HEAD'], check=True)
    tree = os.path.join(dest, 'tree')
    os.makedirs(tree)
    subprocess.run(['tar', '-xf', tar, '-C', tree], check=True)
    os.remove(tar)
    return tree


def inject(tree):
    pkg = os.path.join(tree, 'bitsandbytes', 'bitsandbytes')
    os.makedirs(pkg, exist_ok=True)
    with zipfile.ZipFile(WHEEL) as z:
        n = [x for x in z.namelist() if x.endswith('libbitsandbytes_cpu.dll')][0]
        io.open(os.path.join(pkg, 'libbitsandbytes_cpu.dll'), 'wb').write(z.read(n))
    for d in os.listdir(os.path.join(tree, 'bitsandbytes')):
        if d.endswith('.egg-info'):
            shutil.rmtree(os.path.join(tree, 'bitsandbytes', d))


def build(tree, outdir, epoch):
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    if epoch is None:
        env.pop('SOURCE_DATE_EPOCH', None)
    else:
        env['SOURCE_DATE_EPOCH'] = epoch
    os.makedirs(outdir, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'build', '--sdist', '--outdir', outdir],
                   cwd=os.path.join(tree, 'bitsandbytes'), env=env, check=True,
                   capture_output=True)
    f = [x for x in os.listdir(outdir) if x.endswith('.tar.gz')][0]
    return os.path.join(outdir, f)


def main():
    res = {'epoch': EPOCH, 'runs': []}
    for epoch, label in ((EPOCH, 'fixed'), (EPOCH, 'fixed2'), (None, 'unset'), (None, 'unset2')):
        tmp = tempfile.mkdtemp(prefix='repro_')
        try:
            tree = export(tmp)
            inject(tree)
            out = os.path.join(tmp, 'dist')
            p = build(tree, out, epoch)
            res['runs'].append({'label': label, 'epoch': epoch, 'bytes': os.path.getsize(p),
                                'sha256': sha(p)})
            print('%-7s epoch=%-11s %8d B  %s' % (label, str(epoch), os.path.getsize(p), sha(p)[:16]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    fixed = {r['sha256'] for r in res['runs'] if r['label'].startswith('fixed')}
    unset = {r['sha256'] for r in res['runs'] if r['label'].startswith('unset')}
    res['fixed_reproducible'] = len(fixed) == 1
    res['unset_reproducible'] = len(unset) == 1
    out = r'D:\work\collab\shared\repro_sdist_trial.json'
    io.open(out, 'w', encoding='utf-8', newline='').write(json.dumps(res, ensure_ascii=False, indent=2))
    print('\nfixed 两次一致: %s' % res['fixed_reproducible'])
    print('unset 两次一致: %s' % res['unset_reproducible'])
    print('记录: %s' % out)
    return 0 if res['fixed_reproducible'] else 1


if __name__ == '__main__':
    sys.exit(main())
