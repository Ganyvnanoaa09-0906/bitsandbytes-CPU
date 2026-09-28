"""align_edits.py -- apply the same editorial pass to the English docs that the
Chinese ones already received, so the two sides stay in step.

The Chinese TECH_REPORT.md and TECHNICAL_GUIDE.md were edited to drop author/date
metadata, parenthetical qualifiers in headings, and self-referential framing. Their
English counterparts still carry all of it:

    zh: **范畴**：...            en: **Author** / **Date** / **Scope**: ...
    zh: ### 3.4 线程与精度        en: ### 3.4 Threads & precision (empirical)
    zh: ## 6. 已知约束与问题       en: ## 6. Known Constraints & Pitfalls
    zh: ## 7.2 测试发现与修复      en: ## 7.2 Brute-force review: issues found & fixed

Every replacement is content-addressed and must match exactly once, so a changed
source file fails loudly instead of silently editing the wrong line.

Usage:
    python align_edits.py <docs_dir> [--apply]
"""
import io
import os
import sys

# (file, exact old text, new text)
FIXES = [
    # ---- TECH_REPORT_EN.md -------------------------------------------------
    ("TECH_REPORT_EN.md",
     "**Author**: deepsleep team\n**Date**: 2026-09\n**Scope**: engineering acceleration",
     "**Scope**: engineering acceleration"),
    ("TECH_REPORT_EN.md",
     "### 3.4 Threads & precision (empirical)",
     "### 3.4 Threads & precision"),
    ("TECH_REPORT_EN.md",
     "**Boundary (important)**: this family saves",
     "**Boundary**: this family saves"),
    ("TECH_REPORT_EN.md",
     "**Phase 1: per-operator scheduling = negative (conclusion retained)**",
     "**Phase 1: per-operator scheduling = negative**"),
    ("TECH_REPORT_EN.md",
     "**Phase 2: block-level resident executor (measured viable on this machine)**",
     "**Phase 2: block-level resident executor**"),
    ("TECH_REPORT_EN.md",
     "**Phase 3: CPU/iGPU asynchronous-concurrency squeeze (R9, closed)**",
     "**Phase 3: CPU/iGPU asynchronous concurrency**"),
    ("TECH_REPORT_EN.md",
     "**Crash found on a real LLM (fixed)**: §5's stress test used",
     "**Crash found on a real LLM**: §5's stress test used"),
    ("TECH_REPORT_EN.md",
     "## 7.2 Brute-force review: issues found & fixed",
     "## 7.2 Review: issues found & fixed"),
    # ---- TECHNICAL_GUIDE_EN.md ---------------------------------------------
    ("TECHNICAL_GUIDE_EN.md",
     "> Audience: engineers (developers comfortable reading C++ / PyTorch kernels).\n"
     "> Purpose: explain what this fork **changed relative to bitsandbytes v0.45.1,\n"
     "> why, and how to use it**.\n\n",
     ""),
    ("TECHNICAL_GUIDE_EN.md",
     "**Counter-example measured the same day**: the optimizer's `p` write",
     "**Counter-example**: the optimizer's `p` write"),
    ("TECHNICAL_GUIDE_EN.md",
     "**Key measured results (they shape the architecture)**:",
     "**Conclusion**:"),
    ("TECHNICAL_GUIDE_EN.md",
     "## 6. Known Constraints & Pitfalls",
     "## 6. Known Constraints & Issues"),
]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    apply_changes = "--apply" in sys.argv
    os.chdir(d)

    cache = {}
    applied = missing = 0
    for fname, old, new in FIXES:
        if fname not in cache:
            cache[fname] = io.open(fname, "r", encoding="utf-8", newline="").read()
        text = cache[fname]
        # The files use CRLF. A multi-line pattern written with plain \n will never
        # match, which is what made the first two runs miss on exactly the two
        # multi-line entries. Normalise the SEARCH string to whatever the file
        # actually uses, and normalise the file for the replacement itself so the
        # output keeps its original line endings.
        eol = "\r\n" if "\r\n" in text else "\n"
        old_n = old.replace("\r\n", "\n").replace("\n", eol)
        new_n = new.replace("\r\n", "\n").replace("\n", eol)
        n = text.count(old_n)
        if n != 1:
            print(f"  MISS ({n} match(es), expected 1) {fname}: {old.splitlines()[0][:70]!r}")
            missing += 1
            continue
        cache[fname] = text.replace(old_n, new_n)
        applied += 1
        print(f"  OK   {fname}: {old.splitlines()[0][:66]}")
        print(f"       -> {new.splitlines()[0][:66] if new.strip() else '(removed)'}")

    print()
    print(f"applied {applied}, missed {missing}")
    if missing:
        print("FAIL: some patterns did not match; nothing written")
        return 1

    if apply_changes:
        for fname, text in cache.items():
            tmp = fname + ".tmp"
            with io.open(tmp, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            os.replace(tmp, fname)
            print(f"wrote {fname}")
    else:
        print("\ndry run only; pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
