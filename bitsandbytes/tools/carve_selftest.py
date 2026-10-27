# -*- coding: utf-8 -*-
"""sector_carve 的真实素材自测台（无需坏硬件）。

语料：两张真实 jpg（从 images 里取）+ 由 PIL 生成的 png/gif/pdf + zipfile 生成的真 zip。
镜像：junk 填充 + 文件顺序拼接，记录每个文件的精确偏移。
变体：
  A 干净            → 5 个文件都应 byte-identical 捞回
  B 坏道（jpg 中间清零）→ 该 jpg 允许残缺，但不得影响其它文件
  C 头部被抹（png 签名破坏）→ png 应被跳过
  D 尾部截断（gif 去掉结束符）→ gif 应残缺或跳过，其余正常

判定：按"内容是否 byte-identical"打分，而不是"有没有产出文件"。
用法：python gt1030/carve_selftest2.py [--keep]
"""
from __future__ import annotations
import argparse
import glob
import io
import os
import shutil
import subprocess

from PIL import Image
import zipfile

EXE = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_carve.exe'
WORK = r'F:\weibo_cos\carve_test2'
IMAGES = r'F:\weibo_cos\images'


def corpus():
    """返回 [(name, bytes)]，全部是结构完整的真实文件。"""
    os.makedirs(WORK, exist_ok=True)
    items = []

    # 两张最小真 jpg
    js = sorted(glob.glob(os.path.join(IMAGES, '*.jpg')), key=os.path.getsize)[:2]
    for i, p in enumerate(js):
        items.append(('real%d.jpg' % i, io.open(p, 'rb').read()))

    # png / gif / pdf 由真图转出（结构完整）
    im = Image.open(js[0]).convert('RGB')
    im = im.resize((64, 64))
    for ext, kw in (('png', {}), ('gif', {}), ('pdf', {})):
        p = os.path.join(WORK, 'gen.' + ext)
        im.save(p, **kw)
        items.append(('gen.' + ext, io.open(p, 'rb').read()))

    # 真 zip（含两个条目）
    p = os.path.join(WORK, 'gen.zip')
    with zipfile.ZipFile(p, 'w', zipfile.ZIP_STORED) as z:
        z.writestr('a.txt', b'hello-carve-1' * 20)
        z.writestr('b.bin', bytes(range(256)) * 8)
    items.append(('gen.zip', io.open(p, 'rb').read()))
    return items


def build(variant, items):
    junk = bytes((i * 53 + 7) % 256 for i in range(8192))
    body = bytearray()
    offs = {}
    for name, data in items:
        body += junk
        offs[name] = len(body)
        body += data
    body += junk
    if variant == 'B':
        n, d = items[0]
        s = offs[n] + len(d) // 2
        body[s:s + 64] = b'\x00' * 64
    elif variant == 'C':
        n = 'gen.png'
        s = offs[n]
        body[s:s + 8] = b'\x00' * 8
    elif variant == 'D':
        n = 'gen.gif'
        s = offs[n] + len(dict(items)[n]) - 1
        body[s:s + 1] = b'\x00'
    return bytes(body), offs


def score(out, items):
    """把一个输出目录按「内容是否 byte-identical」打分"""
    byname = dict(items)
    exact, partial = [], []
    for f in sorted(glob.glob(os.path.join(out, '*'))):
        blob = io.open(f, 'rb').read()
        match = [n for n, v in byname.items() if blob == v]
        if match:
            exact.append(match[0])
        else:
            partial.append((os.path.basename(f), len(blob)))
    missing = [n for n in byname if n not in exact]
    return exact, partial, missing


def run(variant, items, keep=False, gui_exe=None):
    d = os.path.join(WORK, variant)
    if os.path.isdir(d) and not keep:
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    img = os.path.join(WORK, 'img_%s.bin' % variant)
    out = os.path.join(d, 'out')
    os.makedirs(out, exist_ok=True)
    data, offs = build(variant, items)
    io.open(img, 'wb').write(data)
    subprocess.run([EXE, img, out], capture_output=True, text=True,
                   encoding='utf-8', errors='replace')
    exact, partial, missing = score(out, items)

    gui = None
    if gui_exe and os.path.isfile(gui_exe):
        gout = os.path.join(d, 'out_gui')
        if os.path.isdir(gout):
            shutil.rmtree(gout, ignore_errors=True)
        os.makedirs(gout, exist_ok=True)
        # GUI 版走同一份内核；这里用它自己的无头自检入口跑同样的镜像
        subprocess.run([gui_exe, '--selftest', img, gout], capture_output=True,
                       encoding='utf-8', errors='replace')
        gexact, gpartial, _gmissing = score(gout, items)
        gui = (gexact, gpartial)
    return exact, partial, missing, len(data), gui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--gui-exe', default=None,
                    help='GUI 版 sector_carve_gui.exe；给了就再跑一遍，校验两个前端结果一致')
    a = ap.parse_args()
    gui = a.gui_exe
    if gui is None:
        cand = os.path.join(os.path.dirname(EXE), 'sector_carve_gui.exe')
        gui = cand if os.path.isfile(cand) else None
    items = corpus()
    print('语料: ' + ', '.join('%s(%dB)' % (n, len(d)) for n, d in items))
    exp = {'A': 6, 'B': 5, 'C': 5, 'D': 5}
    total_ok = 0
    gui_ok = 0
    gui_runs = 0
    for v in 'ABCD':
        exact, partial, missing, n, g = run(v, items, a.keep, gui)
        target = exp[v]
        ok = len(exact) >= target
        total_ok += 1 if ok else 0
        print('\n[%s] 镜像 %.0f KB  完整恢复 %d/%d %s' % (v, n / 1024, len(exact), 6, 'PASS' if ok else 'FAIL'))
        print('    完整: ' + (', '.join(exact) or '(none)'))
        print('    残缺: ' + (', '.join('%s=%dB' % (f, l) for f, l in partial) or '(none)'))
        print('    缺失: ' + (', '.join(missing) or '(none)'))
        if g is not None:
            gui_runs += 1
            gexact, gpartial = g
            same = sorted(gexact) == sorted(exact)
            gui_ok += 1 if same else 0
            print('    GUI 版: 完整 %d/%d %s（%s）' % (
                len(gexact), 6, 'PASS' if same else 'FAIL',
                '与 CLI 一致' if same else '与 CLI 不一致！'))
            if not same:
                print('      GUI 完整集: ' + (', '.join(gexact) or '(none)'))
    print('\n==== %d/4 变体通过 ====' % total_ok)
    if gui_runs:
        print('==== GUI 版与 CLI 结果一致: %d/%d ====' % (gui_ok, gui_runs))
        if gui_ok != gui_runs:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
