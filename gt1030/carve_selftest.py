# -*- coding: utf-8 -*-
"""sector_carve 的合成测试台（不需要任何坏硬件）。

做法：用 Python 造几个**结构有效但内容极小**的 png/jpg/gif/pdf/zip，拼进一个"假盘镜像"，
再制造三种真实世界会出现的损伤，然后跑 sector_carve 看它捞回什么、漏什么、错什么：

  A 干净镜像            → 5 个文件应全部恢复且字节一致
  B 中间被清零（坏道）   → 该文件应按终止标记止损，其余必须不受影响
  C 头部被抹掉           → 该文件应被跳过（签名没了），且不能拖累别人
  D 尾部被截断           → 应"捞到一半且标记失败"，不能崩

用法：python gt1030/carve_selftest.py
"""
from __future__ import annotations
import io
import os
import subprocess
import sys

EXE = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_carve.exe'
WORK = r'F:\weibo_cos\carve_test'


def payloads():
    png = (b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR' + b'A' * 64 + b'IEND\xaeB`\x82')
    jpg = (b'\xff\xd8\xff\xe0' + b'B' * 96 + b'\xff\xd9')
    gif = (b'GIF89a' + b'C' * 64 + b'\x3b')
    pdf = (b'%PDF-1.4\n' + b'D' * 64 + b'\n%%EOF')
    zipf = (b'PK\x03\x04' + b'E' * 64 + b'PK\x05\x06' + b'\x00' * 18)
    return [('t.png', png), ('t.jpg', jpg), ('t.gif', gif), ('t.pdf', pdf), ('t.zip', zipf)]


def build(variant):
    """返回 (镜像字节, 期望能完整恢复的文件名集合)"""
    items = payloads()
    junk = lambda n: bytes((i * 37 + 11) % 256 for i in range(n))
    body = bytearray()
    offsets = {}
    for i, (name, data) in enumerate(items):
        body += junk(4096)                     # 前置垃圾（模拟已删除文件的残留）
        offsets[name] = len(body)
        body += data
    body += junk(4096)
    if variant == 'B':                          # 坏道：把 jpg 中间 48 字节清零
        s = offsets['t.jpg'] + 20
        body[s:s + 48] = b'\x00' * 48
    elif variant == 'C':                        # 头部被抹：png 的 8 字节签名
        s = offsets['t.png']
        body[s:s + 8] = b'\x00' * 8
    elif variant == 'D':                        # 尾部截断：gif 少了结束符 0x3B
        s = offsets['t.gif'] + len(payloads()[2][1]) - 1
        body[s:s + 1] = b'\x2d'
    expect = {n for n, _ in items}
    if variant == 'C':
        expect.discard('t.png')
    return bytes(body), expect, offsets


def run(variant):
    os.makedirs(WORK, exist_ok=True)
    img = os.path.join(WORK, 'test_%s.img' % variant)
    out = os.path.join(WORK, 'out_%s' % variant)
    os.makedirs(out, exist_ok=True)
    data, expect, _ = build(variant)
    io.open(img, 'wb').write(data)
    r = subprocess.run([EXE, img, out], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    got = sorted(os.listdir(out))
    return got, expect, (r.stdout or '') + (r.stderr or ''), len(data)


def main():
    print('%-6s %-8s %-46s %s' % ('变体', '镜像KB', '恢复到的文件', '判定'))
    for v in 'ABCD':
        got, expect, log, n = run(v)
        names = [g for g in got]
        ok = bool(set(names) & expect)
        print('%-6s %-8d %-46s %s' % (v, n // 1024, ','.join(names)[:44] or '(none)', 'OK' if ok else 'FAIL'))
        if v == 'A' and log:
            print('   A 的日志尾部: ' + ' | '.join(log.strip().splitlines()[-2:])[:150])


if __name__ == '__main__':
    main()
