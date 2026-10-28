# -*- coding: utf-8 -*-
"""normalize_sdist.py — 把 sdist 归一化成字节可复现的形式（后处理，不换工具链）。

变异来源只有两类：tar 成员元数据（mtime/uid/gid/uname/gname，以及成员顺序）
与 gzip 头里的 mtime/文件名。两者都在这里消掉：

  * 成员按名字排序，mtime 固定为 epoch，uid/gid=0，uname/gname 置空
  * 用 GNU tar 格式写出（避免 pax 头携带小数 mtime 等不稳定字段）
  * gzip 以 mtime=epoch、无文件名写出

用法：python tools/normalize_sdist.py <in.tar.gz> <out.tar.gz> [--epoch 0]
自检：对同一内容两次构建后归一化，sha256 应相同（本仓库的 repro_trial 会验证）。
"""
import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile


def normalize(src, dst, epoch):
    with tarfile.open(src, 'r:gz') as tin:
        members = sorted(tin.getmembers(), key=lambda m: m.name)
        with open(dst, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=epoch, filename='') as gz:
                with tarfile.open(fileobj=gz, mode='w', format=tarfile.GNU_FORMAT) as tout:
                    for m in members:
                        m.mtime = epoch
                        m.uid = 0
                        m.gid = 0
                        m.uname = ''
                        m.gname = ''
                        data = tin.extractfile(m) if m.isfile() else None
                        tout.addfile(m, data)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--epoch', type=int, default=0)
    a = ap.parse_args()
    normalize(a.src, a.dst, a.epoch)
    print('%s -> %s' % (os.path.basename(a.src), os.path.basename(a.dst)))
    print('  in : %d B  %s' % (os.path.getsize(a.src), sha(a.src)[:16]))
    print('  out: %d B  %s' % (os.path.getsize(a.dst), sha(a.dst)[:16]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
