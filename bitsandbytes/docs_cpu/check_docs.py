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

# A heading number is a dotted section id, optionally with a letter suffix
# (`2A.1`), or a bare integer followed by a separator. `#### 12 个 C 侧导出` is NOT
# section 12: the number is a count followed by a measure word. Without that
# exclusion the zh/en heading comparison reports a phantom section that exists in
# only one language -- and measure words are Chinese, so only the Chinese side is
# affected.
_NUM = r"([0-9]+[A-Z]?(?:\.[0-9]+)*)"
_MEASURE = r"(?!\s*(?:个|项|条|次|张|组|轮|步|种|款|层|类))"
HEAD_RE = re.compile(r"^#{2,4}\s*" + _NUM + _MEASURE + r"(?=[\s.])", re.M)


def section_ids(path):
    return HEAD_RE.findall(io.open(path, encoding="utf-8").read())


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    os.chdir(d)
    names = [n for n in ZH + EN if os.path.isfile(n)]
    print(f"docs dir: {os.path.abspath('.')}")
    print(f"files   : {len(names)}")
    fails = []

    # A heading number is a dotted section id, or a bare integer followed by a
    # separator; see HEAD_RE at module level for why measure words are excluded.
    secs = {}
    for n in names:
        secs[n] = set(section_ids(n))

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
    print("=" * 66)
    print("zh/en section symmetry")
    print("=" * 66)
    for zh, en in zip(ZH, EN):
        if zh not in names or en not in names:
            continue
        a, b = section_ids(zh), section_ids(en)
        ok = a == b
        print(f"  {zh:22s} {len(a):2d} vs {en:26s} {len(b):2d}  {'match' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"{zh} / {en}: section numbering differs")
            only_a = [x for x in a if x not in b]
            only_b = [x for x in b if x not in a]
            if only_a:
                print(f"      only in zh: {only_a[:8]}")
            if only_b:
                print(f"      only in en: {only_b[:8]}")

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
