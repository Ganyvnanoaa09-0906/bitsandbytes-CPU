"""取证扫描：i5 上到底有多少文件被加密了、什么时候被加密的。

判据（不依赖杀软的说法，只看文件本身）：
  · 已知文件头（PNG/JPEG/zip/7z/safetensors/gzip/PDF/EXE/UTF-16 BOM）→ 正常
  · 纯文本（UTF-8/GBK 可解码且可打印字符占比高）→ 正常
  · **都不是，且熵高** → 疑似加密 ✗

另外回答三个关键问题：
  ① 波及范围：是按目录、按扩展名，还是全盘？
  ② 时间分布：被加密文件的 mtime 是否集中在某个时刻？（那时刻就是案发时间）
  ③ **model.safetensors（988MB）还完好么** —— safetensors 的头 8 字节是 JSON 长度，
     能读出合法 JSON 就是好的。

只读，不改任何东西。
"""
import collections
import json
import math
import os
import struct
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MAGIC = {
    b'\x89PNG': 'PNG', b'\xff\xd8\xff': 'JPEG', b'PK\x03\x04': 'ZIP',
    b'7z\xbc\xaf': '7z', b'\x1f\x8b': 'gzip', b'%PDF': 'PDF',
    b'MZ': 'EXE', b'\xff\xfe': 'UTF16', b'\xfe\xff': 'UTF16BE',
    b'GIF8': 'GIF', b'RIFF': 'RIFF', b'\x00\x00\x00\x20ftyp': 'MP4',
    b'ID3': 'MP3', b'OggS': 'OGG', b'\x28\xb5\x2f\xfd': 'zstd',
}
SKIP_EXT = {'.mp4', '.mkv', '.avi', '.mov', '.mp3', '.flac', '.wav',
            '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.ico',
            '.exe', '.dll', '.sys', '.pyc', '.pyd', '.so', '.bin',
            '.zip', '.7z', '.rar', '.gz', '.xz', '.iso', '.vhd'}


def entropy(b):
    if not b:
        return 0.0
    c = collections.Counter(b)
    n = len(b)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def classify(p, size):
    """返回 (类别, 说明)。"""
    try:
        with open(p, 'rb') as f:
            head = f.read(256)
            f.seek(max(0, size // 2))
            mid = f.read(256)
    except Exception as e:
        return 'unreadable', str(e)[:30]
    for m, name in MAGIC.items():
        if head.startswith(m):
            return 'known', name
    if b'\x00' in head[:64] and head[:2] not in (b'\xff\xfe',):
        pass
    # 文本判定：能解码且可打印比例高
    for enc in ('utf-8', 'gbk'):
        try:
            t = head.decode(enc)
            printable = sum(1 for ch in t if ch.isprintable() or ch in '\r\n\t')
            if printable / max(1, len(t)) > 0.92:
                return 'text', enc
        except Exception:
            pass
    h = max(entropy(head), entropy(mid))
    return ('encrypted' if h > 7.5 else 'binary'), 'entropy %.2f' % h


def main():
    roots = sys.argv[1:] or [r'D:\modle', r'D:\deepsleep', r'D:\work_ar',
                             r'D:\Desktop', r'D:\Documents']
    stats = collections.Counter()
    by_dir = collections.defaultdict(collections.Counter)
    times = collections.Counter()
    samples = collections.defaultdict(list)

    for root in roots:
        if not os.path.exists(root):
            print('  跳过（不存在）:', root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if 'node_modules' in dirpath or '__pycache__' in dirpath:
                continue
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                ext = os.path.splitext(fn)[1].lower()
                if ext in SKIP_EXT:
                    stats['skipped_binary_ext'] += 1
                    continue
                try:
                    size = os.path.getsize(fp)
                    mt = os.path.getmtime(fp)
                except Exception:
                    continue
                cat, why = classify(fp, size)
                stats[cat] += 1
                by_dir[dirpath][cat] += 1
                if cat == 'encrypted':
                    times[time.strftime('%Y-%m-%d %H', time.localtime(mt))] += 1
                    if len(samples[dirpath]) < 3:
                        samples[dirpath].append(fn)

    print('=' * 72)
    print('分类统计')
    print('=' * 72)
    for k, v in stats.most_common():
        print('  %-22s %d' % (k, v))

    print('\n有加密文件的目录：')
    for d, c in sorted(by_dir.items(), key=lambda x: -x[1]['encrypted']):
        if c['encrypted']:
            print('  %-52s 加密 %-5d 正常文本 %-5d' % (d[-52:], c['encrypted'], c['text']))
            for s in samples[d]:
                print('        e.g. %s' % s[:60])

    print('\n被加密文件的修改时间分布（案发时间线索）：')
    for t, n in sorted(times.items()):
        print('  %s 时  %d 个' % (t, n))
    if not times:
        print('  （没发现加密文件）')

    print('\n' + '=' * 72)
    print('关键文件检查：model.safetensors 是否完好')
    print('=' * 72)
    mp = r'D:\modle\05b_qwen\model.safetensors'
    if os.path.exists(mp):
        with open(mp, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            blob = f.read(min(n, 4096))
        try:
            hdr = json.loads(blob.decode('utf-8')[:n] if len(blob) >= n
                             else blob.decode('utf-8', 'ignore'))
            print('  ✓ 头部是合法 JSON，%d 字节，%d 个张量 ⇒ 文件完好'
                  % (n, len(hdr)))
            ks = list(hdr)[:3]
            print('    例：%s' % ', '.join(ks))
        except Exception as e:
            print('  ✗ 头部不是合法 JSON（%s）⇒ 这个模型文件也坏了' % str(e)[:50])
            print('    头 8 字节解出的长度 = %d（合理值应是几 KB）' % n)
    else:
        print('  文件不存在:', mp)


if __name__ == '__main__':
    main()
