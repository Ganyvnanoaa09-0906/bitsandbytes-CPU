"""找出真正【无守卫】的 x86 代码 —— 修正判据。

第一版判据错了：它按"条件里是否出现我列的那几个宏"来判定是否 x86 守卫，
于是把 `#ifdef BNB_AVX2_GEMV_4BIT_H`（自定义宏）、`#if defined(GDN_X86)`
这类**合法守卫**全判成"无守卫"，报了 79+35 处，几乎全是误报。
⇒ 只看宏名字的形状，不看语义 —— 又一次同类错误。

正确做法：**不看宏名字**，只看两层信息：
  1. 该行是否处于任意 `#if/#ifdef` 之内（深度 > 0）？
  2. 若是，把**外层条件原文**打出来，由人判断它是否把代码限制在 x86 上。
只有"深度 == 0"（完全在预处理条件之外）才是**无条件编译**的确凿证据。
"""
import os
import re
import sys
import json

SRC = r"D:\work\bitsandbytes-CPU\bitsandbytes\csrc"

X86_PAT = re.compile(
    r"\b(__m128\b|__m128i\b|__m128d\b|__m256\b|__m256i\b|__m256d\b|"
    r"__m512\b|__m512i\b|__m512d\b|"
    r"_mm_[a-z0-9_]+|_mm256_[a-z0-9_]+|_mm512_[a-z0-9_]+)\b"
)
DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def scan(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    stack = []            # [(行号, 条件原文)]
    depth0_hits = []      # 深度 0 的命中 = 无条件编译，确凿缺陷
    func = None

    for i, raw in enumerate(lines, start=1):
        m = DIRECTIVE.match(raw)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ("if", "ifdef", "ifndef"):
                stack.append((i, ("#%s %s" % (kind, rest.strip())).strip()))
            elif kind in ("elif", "else"):
                if stack:
                    stack[-1] = (i, ("#%s %s" % (kind, rest.strip())).strip())
            elif kind == "endif":
                if stack:
                    stack.pop()
            continue

        stripped = raw.strip()
        if stripped.startswith("//"):
            continue
        fm = re.match(r"^\s*(?:static\s+)?(?:inline\s+)?[\w:<>\*&\s]+?\b(\w+)\s*\(", raw)
        if fm and ";" not in raw.split("(")[0]:
            func = fm.group(1)

        if X86_PAT.search(raw) and len(stack) == 0:
            depth0_hits.append({
                "line": i,
                "func": func,
                "code": stripped[:100],
            })
    return depth0_hits


def main():
    all_hits = {}
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith((".cpp", ".c", ".h", ".hpp")):
            continue
        h = scan(os.path.join(SRC, fn))
        if h:
            all_hits[fn] = h

    print("=" * 78)
    print("【确凿缺陷】深度 == 0 的 x86 内在函数用法")
    print("（完全不在任何 #if 内 => 任何架构上都会被编译 => ARM64 必失败）")
    print("=" * 78)
    total = 0
    for fn, hits in all_hits.items():
        byfunc = {}
        for h in hits:
            byfunc.setdefault(h["func"], []).append(h)
        print(f"\n--- {fn}: {len(hits)} 处，涉及 {len(byfunc)} 个函数 ---")
        for fname, hs in byfunc.items():
            print(f"    {fname}()  第 {hs[0]['line']} 行起，{len(hs)} 处")
            print(f"      例: {hs[0]['code']}")
        total += len(hits)

    print("\n" + "=" * 78)
    print(f"确凿缺陷合计: {total} 处")
    print("说明：本判据**只报深度 0**，因此不会像第一版那样把自定义宏守卫误判为无守卫。")
    print("      深度 > 0 但守卫条件可疑的情况不在此列出（需要人读条件原文）。")

    os.makedirs(r"D:\work\cloud_results", exist_ok=True)
    with open(r"D:\work\cloud_results\x86_depth0.json", "w", encoding="utf-8") as f:
        json.dump(all_hits, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
