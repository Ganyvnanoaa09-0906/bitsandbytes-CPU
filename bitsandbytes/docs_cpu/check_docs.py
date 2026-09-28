"""check_docs.py -- structural validation for the user-facing docs.

The docs_cpu set is read by users, so a merge that breaks a code fence or leaves a
dangling cross-reference is a real defect, not a cosmetic one. This checks what a
reader would trip over:

  1. fenced code blocks balance (an odd count swallows the rest of the document)
  2. every `X.md` mention resolves to a file that exists AND, when a section number
     follows, to a section that exists in that file
  3. English docs do not point at the Chinese filenames (a wrong-language link)
  4. tables are intact: every table block has a separator row under its header
  5. no leftover references to the removed FUSED_KERNELS files

Usage:
    python check_docs.py [docs_dir]
"""
import io
import os
import re
import sys

ZH = ["QUICKSTART.md", "TECHNICAL_GUIDE.md", "TECH_REPORT.md"]
EN = ["QUICKSTART_EN.md", "TECHNICAL_GUIDE_EN.md", "TECH_REPORT_EN.md"]
REMOVED = ["FUSED_KERNELS.md", "FUSED_KERNELS_EN.md"]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    os.chdir(d)
    names = [n for n in ZH + EN if os.path.isfile(n)]
    print(f"docs dir: {os.path.abspath('.')}")
    print(f"files   : {len(names)}")
    fails = []

    secs = {}
    for n in names:
        t = io.open(n, encoding="utf-8").read()
        secs[n] = set(re.findall(r"^#{2,4}\s*([0-9]+(?:\.[0-9]+)*)", t, re.M))

    print()
    print("=" * 66)
    print("fences")
    print("=" * 66)
    for n in names:
        lines = io.open(n, encoding="utf-8").read().split("\n")
        f = sum(1 for L in lines if re.match(r"^\s*```", L))
        ok = f % 2 == 0
        print(f"  {n:24s} {f:3d} fence lines  {'balanced' if ok else 'UNBALANCED'}")
        if not ok:
            fails.append(f"{n}: unbalanced code fences")

    print()
    print("=" * 66)
    print("cross-references")
    print("=" * 66)
    bad = 0
    for n in names:
        for i, L in enumerate(io.open(n, encoding="utf-8").read().split("\n"), 1):
            for tgt in names + REMOVED:
                if tgt == n or tgt not in L:
                    continue
                if tgt in REMOVED:
                    print(f"  BAD {n}:{i} references removed file {tgt}")
                    bad += 1
                    fails.append(f"{n}:{i} -> removed {tgt}")
                    continue
                m = re.search(r"(?:§|section\s+)([0-9]+(?:\.[0-9]+)*)", L)
                if m and m.group(1) not in secs[tgt]:
                    print(f"  BAD {n}:{i} -> {tgt} section {m.group(1)} does not exist")
                    bad += 1
                    fails.append(f"{n}:{i} -> {tgt}§{m.group(1)}")
    print(f"  {bad} broken reference(s)")

    print()
    print("=" * 66)
    print("wrong-language links")
    print("=" * 66)
    bad = 0
    for n in EN:
        if n not in names:
            continue
        for i, L in enumerate(io.open(n, encoding="utf-8").read().split("\n"), 1):
            if re.search(r"(?<!_EN)\b(?:TECHNICAL_GUIDE|TECH_REPORT|QUICKSTART)\.md", L) \
                    and "_EN.md" not in L:
                print(f"  BAD {n}:{i}: {L.strip()[:90]}")
                bad += 1
                fails.append(f"{n}:{i} points at a Chinese filename")
    print(f"  {bad} wrong-language link(s)")

    print()
    print("=" * 66)
    print("tables")
    print("=" * 66)
    for n in names:
        lines = io.open(n, encoding="utf-8").read().split("\n")
        blocks = orphans = 0
        i = 0
        while i < len(lines):
            if lines[i].lstrip().startswith("|"):
                blocks += 1
                # a well-formed block has a separator row as its second line
                if i + 1 >= len(lines) or not re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
                    orphans += 1
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    i += 1
            else:
                i += 1
        print(f"  {n:24s} {blocks:3d} table(s), {orphans} without a separator row")
        if orphans:
            fails.append(f"{n}: {orphans} malformed table(s)")

    print()
    if fails:
        print(f"FAIL: {len(fails)} issue(s)")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: all structural checks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
