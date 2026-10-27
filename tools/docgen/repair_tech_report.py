"""Repair TECH_REPORT.md by deleting the two damaged lines, because the evidence says they are residue.

The file had four invalid UTF-8 bytes at the very end, in the pattern

    ...(4/4)。\n  \xbc\x89。\n  \xbc\x89。\n

where \xbc\x89 is a three-byte UTF-8 sequence that lost its leading byte. Two damaged lines.

The repair could have been to guess the missing characters. That is what this report has been arguing
against all day: an unverifiable guess turns a visible defect into an invisible one, and there is no way
to tell afterwards whether the guess was right.

So the question is not "what were those characters" but "was there content there at all". And the
English version answers it. TECH_REPORT_EN.md is the same document translated, it is intact, and it ends
at exactly the same self-test bullet:

    ...`selftest_cpu.c` (4/4).

with nothing after it. The Chinese version has the same content followed by two broken lines that the
English one does not have. So those two lines are not content that got damaged; they are residue from
something that wrote to the end of the file and failed partway through a character.

Deleting them is therefore the repair, and it is verifiable in a way a guess is not: afterwards the file
must decode as strict UTF-8 with no errors, which is a binary property that can be checked.

The deletion is recorded rather than silent. If some later reader finds the English and Chinese versions
differing in length by two lines, this is why.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

P = r"D:\work\bitsandbytes-CPU\docs_cpu\TECH_REPORT.md"
raw = open(P, "rb").read()

print("  === 修复前 ===")
try:
    raw.decode("utf-8")
    print("  ✗ 居然已经是合法 UTF-8 —— 那本脚本不该跑")
    raise SystemExit(1)
except UnicodeDecodeError as e:
    print("  ✓ 确认非法 UTF-8: %s" % str(e)[:80])

# ★ 只在末尾匹配,而且要求那正好是文件的结尾 —— 不做全局替换,避免误伤
TAIL = b"\xbc\x89\xe3\x80\x82\n\xbc\x89\xe3\x80\x82\n"
if not raw.endswith(TAIL):
    print("  ✗ 末尾不是预期的坏字节模式,拒绝盲改")
    print("    实际末尾: %r" % raw[-40:])
    raise SystemExit(1)

print("  ✓ 末尾模式匹配: %r" % raw[-24:])
fixed = raw[: -len(TAIL)]

print()
print("  === 修复后 ===")
print("  删除 %d 字节（那两行残渣）" % len(TAIL))
try:
    text = fixed.decode("utf-8")      # ★ 严格解码,有任何坏字节就抛
    print("  ✓ 现在是【合法 UTF-8】· %d 字节 · %d 字符 · %d 行"
          % (len(fixed), len(text), text.count("\n")))
except UnicodeDecodeError as e:
    print("  ✗ 仍是非法 UTF-8: %s —— 不改文件" % str(e)[:80])
    raise SystemExit(1)

# 末尾现在是干净的
print("  ✓ 新的末尾: %r" % fixed[-60:])

open(P, "wb").write(fixed)

# 复核:重新从磁盘读,确认真的是合法 UTF-8
chk = open(P, "rb").read().decode("utf-8")
print()
print("  === 复核（重新从磁盘读）===")
print("  ✓ %d 字节 · 严格 UTF-8 解码通过 · 末行: %r"
      % (len(chk), chk.rstrip().split("\n")[-1]))
