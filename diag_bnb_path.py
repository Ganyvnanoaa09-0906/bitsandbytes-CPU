"""同一个名字，两种 sys.path 指向，得到完全不同的东西。

  sys.path = D:\\work\\bitsandbytes-CPU              -> 命名空间包（无 __init__.py）
  sys.path = D:\\work\\bitsandbytes-CPU\\bitsandbytes -> 真包（有 __init__.py）

本脚本在**子进程**里分别验证，避免互相污染。
"""
import subprocess
import sys
import json

PATHS = {
    "outer (anime_adiff.py:17 用的这个)": r"D:\work\bitsandbytes-CPU",
    "inner (真正的包所在)": r"D:\work\bitsandbytes-CPU\bitsandbytes",
}

PROBE = r'''
import sys, json, importlib.util as u
sys.path.insert(0, sys.argv[1])
info = {"path": sys.argv[1]}
s = u.find_spec("bitsandbytes")
info["find_spec"] = "None" if s is None else ("namespace" if s.loader is None else "module")
try:
    import bitsandbytes as b
    info["import"] = "OK"
    info["file"] = getattr(b, "__file__", None)
    info["has_nn"] = hasattr(b, "nn")
    info["has_version"] = hasattr(b, "__version__")
    info["version"] = getattr(b, "__version__", None)
except Exception as e:
    info["import"] = f"{type(e).__name__}: {e}"
try:
    from peft.import_utils import is_bnb_available, is_bnb_4bit_available
    info["peft_is_bnb_available"] = is_bnb_available()
    info["peft_is_bnb_4bit_available"] = is_bnb_4bit_available()
except Exception as e:
    info["peft"] = f"{type(e).__name__}: {e}"
print("@@@" + json.dumps(info))
'''

results = {}
for label, p in PATHS.items():
    r = subprocess.run([sys.executable, "-c", PROBE, p],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = [l for l in (r.stdout or "").splitlines() if l.startswith("@@@")]
    if line:
        results[label] = json.loads(line[0][3:])
    else:
        results[label] = {"error": (r.stderr or "")[-400:]}

print("=== 同一份代码，两种 sys.path ===")
for label, info in results.items():
    print(f"\n--- {label}")
    for k, v in info.items():
        print(f"    {k:28s} {v}")

print("\n=== 判定 ===")
outer = results.get("outer (anime_adiff.py:17 用的这个)", {})
inner = results.get("inner (真正的包所在)", {})
checks = [
    ("outer 得到命名空间包（无 __init__.py）", outer.get("find_spec") == "namespace"),
    ("outer 的 bnb 没有 nn", outer.get("has_nn") is False),
    ("outer 让 peft 认为 bnb 不可用或报错",
     outer.get("peft_is_bnb_available") is False or "peft" in outer),
    ("inner 得到真正的包", inner.get("find_spec") == "module"),
    ("inner 的 bnb 有 nn", inner.get("has_nn") is True),
    ("inner 的 bnb 有 __version__", inner.get("has_version") is True),
]
npass = 0
for n, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    npass += bool(ok)
print(f"\n{npass}/{len(checks)} PASS")
sys.exit(0 if npass == len(checks) else 1)
