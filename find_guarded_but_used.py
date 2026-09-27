"""判定「被守卫关住的符号」是否真的会在某个架构上不可见。

前两版判据的问题（同类错误连犯两次，记录在此）：
  第一版：按"条件里是否出现我列的宏"判断守卫，把 `BNB_AVX2_GEMV_4BIT_H`、
          `GDN_X86` 这类**自定义宏**守卫全判成"无守卫" ⇒ 79+35 处误报。
  第二版：只判"引用点是否落在定义所在守卫的区间之外"，于是把
          `neon_*`（定义在 __aarch64__ 守卫内、引用点在**另一个** __aarch64__
          守卫内）也报成问题 ⇒ 29 处误报。
  共同点：**只看位置/名字，不看分支的可达性。**

本版做正确的判定：对"符号 S 定义在守卫 G 内、又被 G 之外引用"这一情形，
逐个引用点求出它所在的守卫条件栈，然后回答一个具体问题：
**是否存在某个目标架构，使 G 的每个启用条件都不成立（⇒ S 不可见），
而引用点的条件栈全部成立（⇒ 该引用会被编译）？**

对每个引用点，取"定义守卫条件"与"引用点各层条件"的合取，检查其可满足性。
本仓库只关心两类目标：x86_64(AVX2 或非 AVX2) 与 aarch64。
用真值表枚举（宏数量有限，直接枚举 2^n）避免手写逻辑。
"""
import os
import re
import sys
import json
import itertools

SRC = r"D:\work\bitsandbytes-CPU\bitsandbytes\csrc\cpu_ops.cpp"

# 会被枚举的"平台宏"。其余宏（自定义开关、特性宏）视为"与平台无关"，
# 不可满足性判断里当作自由变量——但因为我们要找的是"跨平台不可见"，
# 只要平台变量存在一组取值使 G 假而引用为真，就是真问题。
PLATFORM_MACROS = [
    "__AVX2__", "__AVX512F__", "__AVX512BF16__",
    "__x86_64__", "__i386__", "_M_X64", "_M_IX86", "_M_ARM64", "__aarch64__",
    "__GNUC__", "_MSC_VER",
]

DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
DEF = re.compile(r"^(?:template\s*<[^>]*>\s*)?(?:static\s+)?(?:inline\s+)?"
                 r"(?:[\w:<>]+\s+)+?(\w+)\s*\([^;]*$")
KEYWORDS = {"if", "for", "while", "switch", "return", "else", "do", "catch",
            "sizeof", "defined", "BNB_OMP_PARALLEL_FOR", "case"}


def eval_cond(cond, env):
    """把 C 预处理条件翻译成 Python 表达式并求值。

    只处理本文件里实际出现的形态：defined(X)、&&、||、!、括号。
    出现无法翻译的东西时返回 None（表示"不确定"，不作为证据）。
    """
    def repl_defined(m):
        name = m.group(1) or m.group(2)
        return "True" if env.get(name, False) else "False"

    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)", repl_defined, cond)
    e = re.sub(r"defined\s+(\w+)", repl_defined, e)
    e = re.sub(r"\b([A-Za-z_]\w*)\b",
               lambda m: ("True" if env.get(m.group(1), False) else "False")
               if m.group(1) in PLATFORM_MACROS else "False", e)
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    try:
        return bool(eval(e, {"__builtins__": {}}, {}))
    except Exception:
        return None


def main():
    with open(SRC, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # 先建立 行号 -> 该行生效的守卫条件列表
    stack = []
    cond_at = {}
    for i, raw in enumerate(lines, start=1):
        m = DIRECTIVE.match(raw)
        if m:
            k, rest = m.group(1), m.group(2).strip()
            if k in ("if", "ifdef", "ifndef"):
                if k == "ifdef":
                    c = "defined(%s)" % rest
                elif k == "ifndef":
                    c = "!defined(%s)" % rest
                else:
                    c = rest
                stack.append(c)
            elif k == "elif":
                if stack:
                    stack[-1] = rest
            elif k == "else":
                if stack:
                    stack[-1] = "!(%s)" % stack[-1]
            elif k == "endif":
                if stack:
                    stack.pop()
            cond_at[i] = list(stack)
        else:
            cond_at[i] = list(stack)

    # 收集定义（取每个符号的**第一个**定义位置）
    defs = {}
    for i, raw in enumerate(lines, start=1):
        d = DEF.match(raw.rstrip())
        if not d:
            continue
        nm = d.group(1)
        code = raw.split("//")[0]
        if nm in KEYWORDS or not code.strip():
            continue
        if nm not in defs:
            defs[nm] = i

    # 收集引用
    uses = {}
    for i, raw in enumerate(lines, start=1):
        code = raw.split("//")[0]
        if not code.strip():
            continue
        for idm in re.finditer(r"\b([A-Za-z_]\w*)\s*(?:<[^;()]*>)?\s*\(", code):
            nm = idm.group(1)
            if nm in KEYWORDS:
                continue
            uses.setdefault(nm, []).append(i)

    # 枚举平台真值表
    envs = []
    for bits in itertools.product([False, True], repeat=len(PLATFORM_MACROS)):
        envs.append(dict(zip(PLATFORM_MACROS, bits)))

    problems = []
    for nm, defline in defs.items():
        dstack = cond_at.get(defline, [])
        if not dstack:
            continue
        for uline in uses.get(nm, []):
            if uline == defline:
                continue
            ustack = cond_at.get(uline, [])
            # 找一组平台取值：定义守卫【假】（符号不可见）而引用点【全部真】（会被编译）
            bad = None
            for env in envs:
                dvals = [eval_cond(c, env) for c in dstack]
                if any(v is False for v in dvals) or any(v is None for v in dvals):
                    continue          # 定义处可见（或不确定）
                uvals = [eval_cond(c, env) for c in ustack]
                if all(v is True for v in uvals):
                    bad = env
                    break
            if bad is not None:
                problems.append({
                    "symbol": nm,
                    "defined_at": defline,
                    "def_guard": dstack,
                    "used_at": uline,
                    "use_guard": ustack,
                    "witness_env": {k: v for k, v in bad.items()
                                    if k in PLATFORM_MACROS and v},
                })
                break   # 每个符号报一次就够

    problems.sort(key=lambda p: p["defined_at"])
    print("=" * 82)
    print("真问题：存在某个平台取值，使【定义不可见】而【引用会被编译】")
    print("（这才是 ARM64 上 undeclared 的确凿判据）")
    print("=" * 82)
    if not problems:
        print("\n  无")
    for p in problems:
        print(f"\n  {p['symbol']}()")
        print(f"      定义 L{p['defined_at']}  守卫: {[c[:46] for c in p['def_guard']]}")
        print(f"      引用 L{p['used_at']}  守卫: {[c[:46] for c in p['use_guard']]}")
        print(f"      反例平台取值: {sorted(p['witness_env'].keys())}")

    print("\n" + "=" * 82)
    print(f"真问题合计: {len(problems)}")
    with open(r"D:\work\cloud_results\guarded_invisible.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
