# -*- coding: utf-8 -*-
"""test_sdist_identity_adversarial.py — CLI 级对抗测试：身份门禁必须 fail closed。

为什么是 CLI 级：发布门禁的契约是**退出码 + 可否被绕过**，不是内部函数返回值。
所以这里用子进程跑真实入口，断言 (exit code, 输出无 traceback, 关键错误信息)。

覆盖（2026-11-16 对抗审查发现并固化的回归）：
  * 合法布局 PASS；在非 git 目录审计也 PASS（审计不依赖仓库）
  * kernel/vcomp 放在错误目录 → FAIL（此前只按 basename 匹配，可被绕过）
  * 路径穿越 / 绝对路径成员 → FAIL
  * 大写 .PYC / __PyCache__ → FAIL（Windows 大小写不敏感）
  * kernel 是符号链接 → FAIL；重复 kernel → FAIL
  * 空文件 / 随机字节 / gzip 非 tar / 截断 gzip → 干净 FAIL（不是 traceback）
  * 超大 DLL（env 调小上限）→ FAIL；成员数上限 → FAIL
  * --expect-dll 非 64 hex → exit 2；缺 --expect-dll → exit 2；--sdist 与 --dll-source 互斥 → exit 2
  * --json 记录可写出且字段齐全
  * --selftest 自身 16/16

用法：
    python tools/test_sdist_identity_adversarial.py
退出码：0 = 全过；1 = 有失败。
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "check_sdist_identity.py")
PY = sys.executable

KERNEL = b"KERNEL-BYTES-" + b"k" * 200
VCOMP = b"VCOMP-BYTES-" + b"v" * 200
KS = __import__("hashlib").sha256(KERNEL).hexdigest()
VS = __import__("hashlib").sha256(VCOMP).hexdigest()
PKG = "pkg-1.0/bitsandbytes/"
GOOD = [(PKG + "libbitsandbytes_cpu.dll", KERNEL), (PKG + "vcomp140.dll", VCOMP)]
BASE = [
    "--expect-dll",
    KS,
    "--expect-vcomp-dll",
    VS,
    "--dll-source-commit",
    "cdd168fd",
    "--dll-defines=",
    "--dll-toolchain",
    "MSVC",
]


def write_tar(path, files, links=()):
    with tarfile.open(path, "w:gz") as t:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            t.addfile(info)


def run(args, cwd, env_extra=None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("BNB_SDIST_MAX_DLL_MB", None)
    env.pop("BNB_SDIST_MAX_MEMBERS", None)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [PY, GATE] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    if not os.path.isfile(GATE):
        print("[FATAL] 找不到门禁: %s" % GATE, file=sys.stderr)
        return 1
    tmp = tempfile.mkdtemp(prefix="gate_adv_test_")
    failures = []
    checked = 0

    def check(name, args, expect, cwd=None, env=None, contains=None):
        nonlocal checked
        checked += 1
        code, out = run(args, cwd or tmp, env)
        problems = []
        if code != expect:
            problems.append("exit=%d want=%d" % (code, expect))
        if "Traceback" in out:
            problems.append("traceback")
        if contains and contains not in out:
            problems.append("missing %r" % contains)
        if problems:
            failures.append(
                (name, problems, out.strip().splitlines()[-1:] or ["<empty>"])
            )
            print("  [FAIL] %-38s %s" % (name, "; ".join(problems)))
        else:
            print("  [PASS] %-38s exit=%d" % (name, code))

    try:

        def mk(name, files, links=()):
            p = os.path.join(tmp, name)
            write_tar(p, files, links)
            return p

        good = mk("good.tar.gz", GOOD)
        wrong_kernel = mk(
            "wrong_kernel.tar.gz",
            [
                ("pkg-1.0/other/libbitsandbytes_cpu.dll", KERNEL),
                (PKG + "vcomp140.dll", VCOMP),
            ],
        )
        wrong_vcomp = mk(
            "wrong_vcomp.tar.gz",
            [
                (PKG + "libbitsandbytes_cpu.dll", KERNEL),
                ("pkg-1.0/other/vcomp140.dll", VCOMP),
            ],
        )
        traversal = mk(
            "traversal.tar.gz",
            [("../../evil", b"x")] + GOOD,
        )
        absolute = mk("absolute.tar.gz", [("/etc/evil", b"x")] + GOOD)
        uppercase = mk(
            "uppercase.tar.gz",
            GOOD + [(PKG + "a.PYC", b"x"), (PKG + "__PyCache__/b", b"y")],
        )
        symlink = mk(
            "symlink.tar.gz",
            [(PKG + "vcomp140.dll", VCOMP)],
            [(PKG + "libbitsandbytes_cpu.dll", "/etc/passwd")],
        )
        duplicate = mk(
            "duplicate.tar.gz",
            GOOD + [("pkg-1.0/libbitsandbytes_cpu.dll", KERNEL)],
        )
        oversized = mk(
            "oversized.tar.gz",
            [
                (PKG + "libbitsandbytes_cpu.dll", b"K" * (2 << 20)),
                (PKG + "vcomp140.dll", VCOMP),
            ],
        )
        empty = os.path.join(tmp, "empty.bin")
        open(empty, "wb").write(b"")
        junk = os.path.join(tmp, "junk.bin")
        open(junk, "wb").write(os.urandom(4096))
        import gzip

        gztext = os.path.join(tmp, "gztext.gz")
        open(gztext, "wb").write(gzip.compress(b"not a tar"))
        truncated = os.path.join(tmp, "truncated.tar.gz")
        open(truncated, "wb").write(open(good, "rb").read()[:80])

        print("== 合法与形状 ==")
        check(
            "official layout, non-git cwd", ["--sdist", good] + BASE, 0, contains="PASS"
        )
        check(
            "kernel in wrong dir",
            ["--sdist", wrong_kernel] + BASE,
            1,
            contains="错误位置",
        )
        check("vcomp in wrong dir", ["--sdist", wrong_vcomp] + BASE, 1)
        check(
            "path traversal member",
            ["--sdist", traversal] + BASE,
            1,
            contains="路径穿越",
        )
        check(
            "absolute path member", ["--sdist", absolute] + BASE, 1, contains="路径穿越"
        )
        check("uppercase .PYC / __PyCache__", ["--sdist", uppercase] + BASE, 1)
        check("kernel is symlink", ["--sdist", symlink] + BASE, 1)
        check("duplicate kernel", ["--sdist", duplicate] + BASE, 1)
        check(
            "oversized DLL (cap 1 MiB)",
            ["--sdist", oversized] + BASE,
            1,
            env={"BNB_SDIST_MAX_DLL_MB": "1"},
            contains="上限",
        )

        print("== 畸形输入 ==")
        check("empty file", ["--sdist", empty] + BASE, 1, contains="无法解析")
        check("random bytes", ["--sdist", junk] + BASE, 1, contains="无法解析")
        check("gzip of non-tar", ["--sdist", gztext] + BASE, 1, contains="无法解析")
        check("truncated gzip", ["--sdist", truncated] + BASE, 1, contains="无法解析")
        check("directory as --sdist", ["--sdist", tmp] + BASE, 2)

        print("== 参数/用法契约 ==")
        check(
            "bad --expect-dll format", ["--sdist", good, "--expect-dll", "deadbeef"], 2
        )
        check("missing --expect-dll", ["--sdist", good], 2)
        check(
            "--sdist + --dll-source", ["--sdist", good, "--dll-source", good] + BASE, 2
        )
        check("--selftest", ["--selftest"], 0, contains="16/16")

        print("== 记录 ==")
        rec = os.path.join(tmp, "rec.json")
        check("--json record written", ["--sdist", good] + BASE + ["--json", rec], 0)
        try:
            d = json.load(open(rec, encoding="utf-8"))
            need = [
                "mode",
                "audited_artifact_sha256",
                "publish_sha256",
                "kernel_dll_sha256",
                "vcomp_dll_sha256",
                "identity_ok",
                "shape_ok",
                "provenance_attested",
            ]
            missing = [k for k in need if k not in d]
            if missing:
                failures.append(("json record fields", missing, []))
                print("  [FAIL] json record fields missing: %s" % missing)
            else:
                print("  [PASS] json record fields complete")
        except Exception as e:  # noqa: BLE001 - 测试代码：任何解析失败都算失败
            failures.append(("json record parse", [str(e)], []))
            print("  [FAIL] json record parse: %s" % e)

        print("== 成员数上限 ==")
        many = os.path.join(tmp, "many.tar.gz")
        with tarfile.open(many, "w:gz") as t:
            for i in range(50):
                info = tarfile.TarInfo("p/f%03d.txt" % i)
                info.size = 1
                t.addfile(info, io.BytesIO(b"x"))
            for name, data in GOOD:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        check(
            "member cap",
            ["--sdist", many] + BASE,
            1,
            env={"BNB_SDIST_MAX_MEMBERS": "10"},
            contains="上限",
        )
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("FAIL: %d/%d case(s) failed" % (len(failures), checked))
        return 1
    print("PASS: %d/%d adversarial case(s)" % (checked, checked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
