"""Repair the report's corrupted raw-log blocks, using the evidence files as the source of truth.

What went wrong: when appendix A.10/A.11/A.13 were assembled, several logs were read as *bytes* and
decoded as UTF-8 with errors='replace' instead of being decoded from UTF-16LE. A UTF-16LE ASCII log is
half NUL bytes, so the report ended up with lines like "\\x00[\\x00v\\x002\\x00]\\x00" and U+FFFD runs.

Every corrupted block has a matching evidence file:
    7 blocks under D:\\work\\research\\evidence\\raw\\<name>.log   (UTF-16LE with BOM)
    4 blocks under D:\\work\\research\\evidence\\raw_outputs\\<name>.txt  (UTF-16LE with BOM)
plus 2 path lines in section 7.75.4 whose "\\b" and "\\a" were interpreted as escapes; the intended
text is in write_775.py lines 75-76.

Nothing is reconstructed by hand: each block is replaced by the decoded evidence file, and the script
refuses to write if a block it expected to be corrupt is not, or if any block ends up still dirty.
"""
from __future__ import annotations

import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
RAW = r'D:\work\research\evidence\raw'
OUT = r'D:\work\research\evidence\raw_outputs'
BACKUP = r'D:\work\research\evidence\CPU_FORGE_TECHNICAL_REPORT.before_repair_20261017.md'
GEN_775 = r'D:\work\bitsandbytes-CPU\write_775.py'

LOG_BLOCKS = [
    ('curriculum_B1.log', os.path.join(RAW, 'curriculum_B1.log')),
    ('curriculum_B2.log', os.path.join(RAW, 'curriculum_B2.log')),
    ('maskgit_causal.log', os.path.join(RAW, 'maskgit_causal.log')),
    ('maskgit_sample.log', os.path.join(RAW, 'maskgit_sample.log')),
    ('maskgit_train.log', os.path.join(RAW, 'maskgit_train.log')),
    ('night_ns.log', os.path.join(RAW, 'night_ns.log')),
    ('night_nsasync.log', os.path.join(RAW, 'night_nsasync.log')),
    ('night_tk.log', os.path.join(RAW, 'night_tk.log')),
    ('night_ttl.log', os.path.join(RAW, 'night_ttl.log')),
]

A13_BLOCKS = [
    ('### A.13.1 CPU 算术上限（重测，空闲机器）', os.path.join(OUT, 'cpu_fma_headroom.txt')),
    ('### A.13.2 内存带宽（重测，空闲机器）', os.path.join(OUT, 'measure_real_bandwidth.txt')),
    ('### A.13.3 训练真实形状的 matmul', os.path.join(OUT, 'sweep_training_shapes.txt')),
    ('### A.13.4 核显 v4 内核（现行输出，与 7.22.4 的 300 不符）', os.path.join(OUT, 'igpu_gemm_v4.txt')),
]

DIRTY = ('\x00', '\ufffd')


def read_text(path):
    b = open(path, 'rb').read()
    if b[:2] == b'\xff\xfe':
        s = b.decode('utf-16-le')[1:]      # drop the BOM char decoded from ff fe
    elif b[:2] == b'\xef\xbb':
        s = b.decode('utf-8-sig')
    else:
        s = b.decode('utf-8')
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = s.strip('\n')
    if any(ln.strip().startswith('```') for ln in s.split('\n')):
        raise SystemExit('ABORT: evidence %s contains a fence line' % path)
    return s


def is_dirty(s):
    return any(ch in s for ch in DIRTY)


def find_line(lines, needle, start=0, exact=True):
    for i in range(start, len(lines)):
        if exact and lines[i] == needle:
            return i
        if (not exact) and lines[i].startswith(needle):
            return i
    return -1


def main(dry):
    text = open(REPORT, encoding='utf-8').read()
    lines = text.split('\n')
    print('report: %d chars, %d lines' % (len(text), len(lines)))

    plan = []   # (start, end, replacement_lines, label)

    for heading, src in LOG_BLOCKS:
        # The same heading can occur twice: once inside the clean A.10 mega-fence and once as its
        # own block in A.11. Pick the occurrence whose fenced block is actually dirty.
        found = None
        h = 0
        while True:
            h = find_line(lines, '### ' + heading, h)
            if h < 0:
                break
            i_open = find_line(lines, '```', h + 1)
            if i_open < 0:
                break
            i_close = find_line(lines, '```', i_open + 1)
            if i_close < 0:
                break
            block = '\n'.join(lines[i_open + 1:i_close])
            if is_dirty(block):
                found = (h, i_open, i_close, block)
                break
            h += 1
        if found is None:
            print('SKIP  %s: no dirty fenced block found' % heading)
            continue
        h, i_open, i_close, block = found
        new = read_text(src).split('\n')
        plan.append((i_open + 1, i_close, new, heading))
        print('LOG   %-24s heading line %d · old lines %d..%d (%d dirty chars) -> %d lines from %s'
              % (heading, h + 1, i_open + 2, i_close,
                 sum(block.count(c) for c in DIRTY), len(new), os.path.basename(src)))

    for heading, src in A13_BLOCKS:
        h = find_line(lines, heading)
        if h < 0:
            print('SKIP  missing heading %r' % heading)
            continue
        i_open = find_line(lines, '```', h + 1)
        i_close = find_line(lines, '```', i_open + 1)
        # after the command block: everything up to the next "### A.13" heading is the broken part
        nxt = -1
        for i in range(i_close + 1, len(lines)):
            if lines[i].startswith('### A.13'):
                nxt = i
                break
        if nxt < 0:
            nxt = len(lines)
        seg = '\n'.join(lines[i_close + 1:nxt])
        if not is_dirty(seg):
            print('SKIP  %s is already clean' % heading)
            continue
        body = read_text(src).split('\n')
        new = ['', '```'] + body + ['```', '']
        plan.append((i_close + 1, nxt, new, heading))
        print('A13   %-40s old lines %d..%d (%d dirty chars) -> %d lines from %s'
              % (heading[4:24], i_close + 2, nxt, sum(seg.count(c) for c in DIRTY), len(new), os.path.basename(src)))

    # Section 7.75.4: two path lines that went through escape interpretation.
    gen_lines = open(GEN_775, encoding='utf-8').read().split('\n')
    good_paths = [ln for ln in gen_lines if ln.startswith('D:\\work\\bnb-4bitopt')]
    if len(good_paths) != 2:
        print('ABORT: write_775.py does not contain exactly two intended path lines (found %d)' % len(good_paths))
        return 1
    for i in range(len(lines) - 1):
        if lines[i].startswith('D:\\work') and ('\x08' in lines[i] or '\x07' in lines[i]):
            if '\x08' in lines[i + 1] or '\x07' in lines[i + 1]:
                plan.append((i, i + 2, good_paths, '7.75.4 path lines'))
                print('PATH  lines %d..%d -> %r / %r' % (i + 1, i + 2, good_paths[0][:40], good_paths[1][:40]))
                break

    if not plan:
        print('nothing to repair')
        return 0

    for start, end, new, label in sorted(plan, key=lambda p: -p[0]):
        lines[start:end] = new

    out = '\n'.join(lines)
    left = sum(out.count(c) for c in DIRTY)
    print()
    print('result: %d chars, %d lines, %d dirty chars left' % (len(out), len(lines), left))
    if left:
        print('remaining dirty lines:')
        shown = 0
        for n, ln in enumerate(lines):
            if any(ch in ln for ch in DIRTY):
                vis = ''.join(c if c.isprintable() else '<%02x>' % ord(c) for c in ln)
                print('   %5d | %s' % (n + 1, vis[:140]))
                shown += 1
                if shown >= 12:
                    break
        print('ABORT: dirty characters remain')
        return 1
    if dry:
        print('(dry run: not written)')
        return 0

    shutil.copy2(REPORT, BACKUP)
    print('backup: %s' % BACKUP)
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        f.write(out)
    print('written: %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main('--dry' in sys.argv))
