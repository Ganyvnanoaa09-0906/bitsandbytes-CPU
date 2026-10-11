"""Enumerate the fork's actual public surface, so the reference table is not guessed.

Prints what the package exports, what each class's constructor takes, and what the
fork-specific modules add on top of upstream. Everything here comes from the installed
code, so a name that appears in the table is a name that exists.
"""
import inspect
import os
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU\bitsandbytes")

import bitsandbytes as bnb  # noqa: E402


def sig(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:  # noqa: BLE001
        return "(?)"


def short(doc):
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line[:88]
    return ""


print("=" * 78)
print("bitsandbytes/__init__.py exports")
print("=" * 78)
print("  " + ", ".join(sorted(n for n in dir(bnb) if not n.startswith("_"))))

print()
print("=" * 78)
print("bitsandbytes.nn")
print("=" * 78)
import bitsandbytes.nn as bnn  # noqa: E402
for n in sorted(dir(bnn)):
    if n.startswith("_"):
        continue
    o = getattr(bnn, n)
    if inspect.isclass(o):
        print(f"  {n}{sig(o.__init__)}")
        d = short(o.__doc__)
        if d:
            print(f"      {d}")

print()
print("=" * 78)
print("bitsandbytes.optim")
print("=" * 78)
import bitsandbytes.optim as bo  # noqa: E402
names = sorted(n for n in dir(bo) if not n.startswith("_"))
for n in names:
    o = getattr(bo, n)
    if inspect.isclass(o) and "bit" in n.lower():
        s = sig(o.__init__)
        # keep it readable
        s = s.replace("self, ", "").replace("self", "")
        print(f"  {n}{s[:150]}")

print()
print("=" * 78)
print("bitsandbytes.functional  (public functions)")
print("=" * 78)
import bitsandbytes.functional as bf  # noqa: E402
for n in sorted(dir(bf)):
    if n.startswith("_"):
        continue
    o = getattr(bf, n)
    if inspect.isfunction(o) and getattr(o, "__module__", "").startswith("bitsandbytes"):
        print(f"  {n}{sig(o)}")

print()
print("=" * 78)
print("fork-specific modules")
print("=" * 78)
for mod in ("gdn_cpu",):
    try:
        m = __import__(f"bitsandbytes.{mod}", fromlist=["x"])
        print(f"  bitsandbytes.{mod}:")
        for n in sorted(dir(m)):
            if n.startswith("_"):
                continue
            o = getattr(m, n)
            kind = "class" if inspect.isclass(o) else (
                "def" if inspect.isfunction(o) else "obj")
            if getattr(o, "__module__", "").startswith("bitsandbytes") or kind != "obj":
                print(f"      {kind:5s} {n}{sig(o) if kind != 'obj' else ''}"[:130])
    except Exception as exc:  # noqa: BLE001
        print(f"  bitsandbytes.{mod}: import failed {exc}")

print()
print("=" * 78)
print("repo-level scripts with a CLI")
print("=" * 78)
root = r"D:\work\bitsandbytes-CPU\bitsandbytes"
for fn in sorted(os.listdir(root)):
    if fn.endswith(".py"):
        p = os.path.join(root, fn)
        try:
            src = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if "argparse" in src or "sys.argv" in src:
            print(f"  {fn}")
