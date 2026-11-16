# -*- coding: utf-8 -*-
"""test_check_docs_adversarial.py — 文档结构门禁必须 fail closed。

背景（2026-11-16 对抗审查）：check_docs.py 原来只检查"存在的文件"，所以
删掉任意或全部英文文档后它仍打印 PASS —— Gate-2 第 6 项可被"删文档"绕过。
现在要求 ZH+EN 六份齐全，缺一即 FAIL。

用例（复制 docs_cpu 到临时目录后定向破坏）：
  * 完整文档 → PASS
  * 删 1 份 EN / 删全部 EN → FAIL 且报"缺"
  * 空目录 → FAIL
  * 坏 fence / 坏表格 / 引用已删除文件 / 章节号不对称 → FAIL

用法：python tools/test_check_docs_adversarial.py   （0=PASS）
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "bitsandbytes", "docs_cpu")
CHECK = os.path.join(DOCS, "check_docs.py")
ZH = ["QUICKSTART.md", "TECHNICAL_GUIDE.md", "TECH_REPORT.md"]
EN = ["QUICKSTART_EN.md", "TECHNICAL_GUIDE_EN.md", "TECH_REPORT_EN.md"]


def run(d):
    r = subprocess.run(
        [sys.executable, CHECK, d],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def copy_docs(d):
    for n in ZH + EN:
        shutil.copy2(os.path.join(DOCS, n), os.path.join(d, n))


def main() -> int:
    if not os.path.isfile(CHECK):
        print("[FATAL] 找不到 check_docs.py: %s" % CHECK, file=sys.stderr)
        return 1
    failures = []
    checked = 0

    def case(name, mutate, expect, contains=None):
        nonlocal checked
        checked += 1
        d = tempfile.mkdtemp(prefix="docs_adv_")
        try:
            copy_docs(d)
            mutate(d)
            code, out = run(d)
            problems = []
            if code != expect:
                problems.append("exit=%d want=%d" % (code, expect))
            if contains and contains not in out:
                problems.append("missing %r" % contains)
            if problems:
                failures.append((name, problems))
                print("  [FAIL] %-34s %s" % (name, "; ".join(problems)))
            else:
                print("  [PASS] %-34s exit=%d" % (name, code))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def del_en(d):
        os.remove(os.path.join(d, EN[0]))

    def del_all_en(d):
        for n in EN:
            os.remove(os.path.join(d, n))

    def bad_fence(d):
        p = os.path.join(d, ZH[0])
        s = io.open(p, encoding="utf-8").read()
        io.open(p, "w", encoding="utf-8").write(s + "\n```\n")

    def bad_table(d):
        p = os.path.join(d, ZH[1])
        s = io.open(p, encoding="utf-8").read()
        io.open(p, "w", encoding="utf-8").write(s + "\n| a | b |\n| c | d |\n")

    def removed_ref(d):
        p = os.path.join(d, ZH[0])
        s = io.open(p, encoding="utf-8").read()
        io.open(p, "w", encoding="utf-8").write(s + "\n见 FUSED_KERNELS.md\n")

    def sec_mismatch(d):
        p = os.path.join(d, ZH[2])
        s = io.open(p, encoding="utf-8").read()
        io.open(p, "w", encoding="utf-8").write(s + "\n## 99.99 phantom section\n")

    case("complete ZH+EN", lambda d: None, 0, contains="PASS")
    case("one EN missing", del_en, 1, contains="缺")
    case("all EN missing", del_all_en, 1, contains="缺")
    case("bad fence", bad_fence, 1)
    case("bad table", bad_table, 1)
    case("reference to removed file", removed_ref, 1)
    case("section numbering mismatch", sec_mismatch, 1)

    checked += 1
    empty = tempfile.mkdtemp(prefix="docs_adv_empty_")
    try:
        code, out = run(empty)
        if code != 1:
            failures.append(("empty dir", ["exit=%d want=1" % code, out[-80:]]))
            print("  [FAIL] %-34s exit=%d" % ("empty dir", code))
        else:
            print("  [PASS] %-34s exit=1" % "empty dir")
    finally:
        shutil.rmtree(empty, ignore_errors=True)

    print()
    if failures:
        print("FAIL: %d/%d case(s)" % (len(failures), checked))
        return 1
    print("PASS: %d/%d adversarial doc case(s)" % (checked, checked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
