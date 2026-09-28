r"""check_bundle_includes.py -- does the bundle contain every local header the
build will ask for?

The first Termux bundle was assembled by hand and went out without
csrc/common.h, so the ARM64 build died on:

    csrc/cpu_ops.h:4:10: fatal error: 'common.h' file not found

...while every kernel in it was fine. That is a packaging defect that looks like
a code defect, and it wastes a whole round trip to a device. This checks it
locally instead: walk every #include "..." in the CPU build's sources and assert
the named file exists in the bundle's csrc.

Only local (quoted) includes are checked -- those are the ones the bundle has to
supply. Angle-bracket includes come from the toolchain.

Usage:
    python check_bundle_includes.py <bundle_root>
where <bundle_root> is the extracted bnb/ directory (contains csrc/).
"""
import os
import re
import sys

RE_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)

SOURCES = ["cpu_ops.cpp", "cpu_gdn.cpp", "pythonInterface.cpp", "cpu_ops.h",
           "selftest_cpu.c"]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = os.path.abspath(sys.argv[1])
    csrc = os.path.join(root, "csrc")
    if not os.path.isdir(csrc):
        print(f"FATAL: no csrc/ under {root}")
        return 2

    available = set(os.listdir(csrc))
    print(f"bundle csrc: {len(available)} entries")
    print()

    problems = []
    checked = set()
    for src in SOURCES:
        # selftest_cpu.c lives at the bundle root, the rest in csrc/
        p = os.path.join(csrc, src)
        if not os.path.isfile(p):
            p = os.path.join(root, src)
        if not os.path.isfile(p):
            problems.append((src, "<source file itself missing from bundle>"))
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for inc in RE_INCLUDE.findall(text):
            # normalise ./ and subdir forms
            base = os.path.basename(inc)
            if base in checked:
                continue
            checked.add(base)
            if base in available:
                print(f"  [OK]   {inc:28s} (from {src})")
            else:
                print(f"  [MISS] {inc:28s} (from {src})")
                problems.append((src, inc))

    # Also verify the transitive case: a header that includes another header.
    for h in sorted(available):
        if not h.endswith((".h", ".cuh")):
            continue
        with open(os.path.join(csrc, h), "r", encoding="utf-8", errors="replace") as fh:
            for inc in RE_INCLUDE.findall(fh.read()):
                base = os.path.basename(inc)
                if base not in available:
                    print(f"  [MISS] {inc:28s} (from {h}, transitive)")
                    problems.append((h, inc))

    print()
    if problems:
        print(f"FAIL: {len(problems)} include(s) cannot be satisfied from the bundle:")
        for src, inc in problems:
            print(f"  {src} -> {inc}")
        print()
        print("Add them to the bundle. Do NOT hand-pick files out of csrc: copy the")
        print("directory and let the build decide what it needs.")
        return 1
    print(f"PASS: all {len(checked)} local include(s) resolvable from the bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
