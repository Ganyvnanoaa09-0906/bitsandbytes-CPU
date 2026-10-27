"""Locate control-character / U+FFFD corruption in the report, with byte offsets and line numbers.

The report is strict UTF-8, so this is not an encoding failure: it is *content* that was
written through a wrong decode somewhere upstream (escapes collapsed, UTF-16 pasted as bytes).
Repair needs evidence, so this only inventories.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PATH = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

with open(PATH, 'rb') as f:
    raw = f.read()
text = raw.decode('utf-8')          # strict: raises if this is ever not UTF-8

# Byte offset of each character index
offsets = []
i = 0
for ch in text:
    offsets.append(i)
    i += len(ch.encode('utf-8'))

bad_chars = {'\x00': 'NUL', '\x01': 'SOH', '\x02': 'STX', '\x03': 'ETX', '\x04': 'EOT',
             '\x05': 'ENQ', '\x06': 'ACK', '\x07': 'BEL', '\x08': 'BS', '\x0b': 'VT',
             '\x0c': 'FF', '\x0e': 'SO', '\x0f': 'SI', '\x10': 'DLE', '\x11': 'DC1',
             '\x12': 'DC2', '\x13': 'DC3', '\x14': 'DC4', '\x15': 'NAK', '\x16': 'SYN',
             '\x17': 'ETB', '\x18': 'CAN', '\x19': 'EM', '\x1a': 'SUB', '\x1b': 'ESC',
             '\x1c': 'FS', '\x1d': 'GS', '\x1e': 'RS', '\x1f': 'US', '\ufffd': 'FFFD'}

# Line starts
line_starts = [0]
for idx, ch in enumerate(text):
    if ch == '\n':
        line_starts.append(idx + 1)


def lineno(pos):
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


hits = []
for pos, ch in enumerate(text):
    if ch in bad_chars:
        hits.append((pos, ch))

print('file bytes: %d · chars: %d · lines: %d' % (len(raw), len(text), text.count('\n') + 1))
print('bad characters: %d' % len(hits))

# Group hits into regions separated by > 400 chars without a hit
regions = []
cur = []
last = None
for pos, ch in hits:
    if last is not None and pos - last > 400:
        regions.append(cur)
        cur = []
    cur.append((pos, ch))
    last = pos
if cur:
    regions.append(cur)

print('corrupt regions (gap > 400 chars): %d' % len(regions))
print()
for ri, reg in enumerate(regions):
    start, end = reg[0][0], reg[-1][0]
    kinds = {}
    for pos, ch in reg:
        kinds[bad_chars[ch]] = kinds.get(bad_chars[ch], 0) + 1
    l1, l2 = lineno(start), lineno(end)
    print('region %d  chars %d..%d  bytes %d..%d  lines %d..%d  n=%d  %s'
          % (ri + 1, start, end, offsets[start], offsets[end] + 1, l1, l2, len(reg),
             ' '.join('%s:%d' % kv for kv in sorted(kinds.items(), key=lambda kv: -kv[1]))))
    # context: the line containing the region start
    ls = line_starts[l1 - 1]
    le = text.find('\n', ls)
    line = text[ls:le if le >= 0 else len(text)]
    printable = ''.join(c if (c.isprintable() or c == '\t') else '<%s>' % bad_chars.get(c, '?') for c in line)
    print('    start line: %s' % printable[:160])
print()

# nearest heading above each region, to say which section it is in
print('=== section attribution ===')
import re
heading_re = re.compile(r'^#{1,4} .*$', re.M)
headings = [(m.start(), m.group(0)) for m in heading_re.finditer(text)]
for ri, reg in enumerate(regions):
    start = reg[0][0]
    above = None
    for hpos, htext in headings:
        if hpos <= start:
            above = (hpos, htext)
        else:
            break
    print('region %d -> %s' % (ri + 1, above[1] if above else '(none)'))
