"""针对 D:\modle\05b_qwen 那几个"密文 txt"的专项取证。

要回答的问题：
  ① 它们的 mtime 是否相同？（相同 ⇒ 同一时刻被某个程序处理过）
  ② 大小是否对齐到 16 / 512 字节？（块密码或扇区对齐的痕迹）
  ③ 几个文件的头部是否有公共前缀？（同 key 复用）
  ④ **i5 上有没有同名或同内容但可读的副本？**（有 ⇒ 证明原本是文本）
  ⑤ 同目录下还有哪些文件是【可读】的？（看什么活下来了）
  ⑥ 这几个文件里是否藏了别的格式（zip/7z/gpg/age/openssl 的头在偏移 0 之外？）

只读。
"""
import hashlib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

D = r'D:\modle\05b_qwen'
TARGETS = ['training_lora_log.txt', 'training_spy_log.txt',
           'requirements_实验环境.txt', '苏白粥_阅读摘要.txt',
           '校花学姐从无绯闻，直到我上大学.txt']


def entropy(b):
    import collections
    import math
    if not b:
        return 0.0
    c = collections.Counter(b)
    n = len(b)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


print('=' * 74)
print('① 大小 / 对齐 / 时间 / 头部')
print('=' * 74)
print('%-30s %12s %6s %6s  %-19s %s' % ('文件', '字节', '%16', '%512', '修改时间', '头16字节'))
meta = {}
for fn in TARGETS:
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        print('%-30s （不存在）' % fn[:30])
        continue
    sz = os.path.getsize(p)
    mt = os.path.getmtime(p)
    import time as _t
    with open(p, 'rb') as f:
        head = f.read(64)
    meta[fn] = (sz, mt, head[:16])
    print('%-30s %12d %6d %6d  %-19s %s'
          % (fn[:30], sz, sz % 16, sz % 512,
             _t.strftime('%Y-%m-%d %H:%M:%S', _t.localtime(mt)),
             head[:16].hex(' ')))

print('\n=== mtime 是否同一时刻 ===')
ms = sorted(set(int(m) for (_s, m, _h) in meta.values()))
if len(ms) == 1:
    print('  ★ 全部相同 ⇒ 同一时刻被某个程序处理过 ⇒ 事后事件')
else:
    import time as _t
    print('  不同（跨度 %d 秒）：' % (ms[-1] - ms[0]))
    for m in ms:
        print('     %s' % _t.strftime('%Y-%m-%d %H:%M:%S', _t.localtime(m)))
    print('  ⇒ 分散写入 ⇒ 更像【训练时本来就写成这样】，不是被事后加密')

print('\n=== 头部公共前缀 ===')
heads = [h for (_s, _m, h) in meta.values()]
common = 0
for i in range(min(len(h) for h in heads)):
    if len({h[i] for h in heads}) == 1:
        common += 1
    else:
        break
print('  最长公共前缀 = %d 字节 %s' % (common, '（无 ⇒ 各自独立随机/独立 IV）' if common == 0 else ''))

print('\n=== 熵（前 4KB）===')
for fn in TARGETS:
    p = os.path.join(D, fn)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            b = f.read(4096)
        print('  %-30s %.3f  (8.0 = 完全随机)' % (fn[:30], entropy(b)))

print('\n' + '=' * 74)
print('② 同目录下的文件：哪些可读？')
print('=' * 74)
for fn in sorted(os.listdir(D)):
    p = os.path.join(D, fn)
    if os.path.isdir(p):
        print('  [目录] %s' % fn)
        continue
    try:
        with open(p, 'rb') as f:
            b = f.read(2048)
        e = entropy(b)
        kind = '密文?' if e > 7.5 else ('文本' if all(
            (32 <= c < 127) or c in (9, 10, 13) or c > 127 for c in b[:500]) else '二进制')
        print('  %-46s %10d 熵%.2f  %s' % (fn[:46], os.path.getsize(p), e, kind))
    except Exception as ex:
        print('  %-46s 读失败 %s' % (fn[:46], str(ex)[:30]))

print('\n' + '=' * 74)
print('③ 全盘找同名或同大小的副本（可读的 = 原本是文本的证据）')
print('=' * 74)
target_names = {fn.lower() for fn in TARGETS}
target_sizes = {s for (s, _m, _h) in meta.values()}
hits = 0
for root in (r'D:\\', r'C:\Users\GanYv'):
    for dirpath, dirnames, filenames in os.walk(root):
        if 'node_modules' in dirpath or '__pycache__' in dirpath:
            continue
        for fn in filenames:
            if fn.lower() in target_names or os.path.splitext(fn)[0].lower() in \
                    {os.path.splitext(t)[0].lower() for t in TARGETS}:
                if os.path.join(dirpath, fn) == os.path.join(D, fn):
                    continue
                print('  同名: %s' % os.path.join(dirpath, fn)[:100])
                hits += 1
                if hits > 12:
                    break
        if hits > 12:
            break
if not hits:
    print('  没找到同名副本')

print('\n' + '=' * 74)
print('④ 偏移 0 之外是否藏了已知格式（加密容器常见）')
print('=' * 74)
SIGS = {b'PK\x03\x04': 'ZIP', b'7z\xbc\xaf': '7z', b'\x1f\x8b': 'gzip',
        b'-----BEGIN': 'PEM/PGP', b'age-encryption': 'age',
        b'Salted__': 'openssl', b'Rar!': 'RAR', b'BZh': 'bzip2',
        b'\xfd7zXZ': 'xz', b'\x28\xb5\x2f\xfd': 'zstd'}
for fn in TARGETS:
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        continue
    with open(p, 'rb') as f:
        data = f.read(1 << 20)
    found = []
    for sig, name in SIGS.items():
        i = data.find(sig)
        if i >= 0:
            found.append('%s@%d' % (name, i))
    print('  %-30s %s' % (fn[:30], ', '.join(found) if found else '无已知格式头'))
