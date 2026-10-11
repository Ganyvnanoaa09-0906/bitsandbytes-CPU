"""safetensors 头校验（修正版）。

上一版的 bug：只读 4096 字节却拿头 8 字节声明的长度去解析 JSON。
头长度实际是 32280 ⇒ 被截断 ⇒ 误报 "Unterminated string"。

正确做法：先读 8 字节拿长度 n，再读满 n 字节。
判据：JSON 能解析 + 每个张量的 data_offsets 都落在文件大小内 ⇒ 文件完好。
"""
import json
import os
import struct
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for p in sys.argv[1:]:
    if not os.path.exists(p):
        print('不存在:', p)
        continue
    size = os.path.getsize(p)
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        blob = f.read(n)                    # ← 读满，不再截断
    print('=' * 66)
    print(p)
    print('  文件 %d 字节 | 头声明 %d 字节' % (size, n))
    if n > 100 * 1024 * 1024:
        print('  ✗ 头长度不合理（>100MB）⇒ 文件坏了')
        continue
    try:
        hdr = json.loads(blob.decode('utf-8'))
    except Exception as e:
        print('  ✗ 头 JSON 解析失败: %s' % str(e)[:80])
        print('    头 64 字节: %s' % blob[:64])
        continue
    tensors = {k: v for k, v in hdr.items() if k != '__metadata__'}
    data_start = 8 + n
    bad = []
    total = 0
    for k, v in tensors.items():
        try:
            a, b = v['data_offsets']
        except Exception:
            bad.append((k, 'no offsets'))
            continue
        total += b - a
        if data_start + b > size:
            bad.append((k, 'offset %d 超出文件' % (data_start + b)))
    print('  ✓ 头部是合法 JSON：%d 个张量' % len(tensors))
    print('    数据类型: %s' % sorted({v.get('dtype', '?') for v in tensors.values()}))
    print('    张量数据合计 %d 字节 (%.2f GB)，加头部共 %d 字节'
          % (total, total / 2**30, data_start + total))
    if bad:
        print('  ✗ %d 个张量的偏移越界（前 3 个）：' % len(bad))
        for k, why in bad[:3]:
            print('      %s: %s' % (k, why))
    else:
        print('  ✓ 所有张量偏移都在文件范围内 ⇒ **文件完好**')
