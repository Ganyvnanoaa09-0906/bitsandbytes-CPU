# -*- coding: utf-8 -*-
"""test_normalize_limits.py — 钉住 normalize_sdist 的"归一化边界"。

为什么需要这个测试：发布身份（dev10 的归一化 sha256 `b85cf8086d36fb94…`、
解压 tar `7d5d95a1e6eb86f8…`）是在当前归一化实现下记录的。归一化器**改一点，
记录就全失效**。所以这里同时钉住两件事：

1. 它确实消除的变异：mtime / uid+gid / uname+gname / 成员顺序 / pax 头 / 幂等性 /
   symlink 保留 / 重复成员确定性 / gzip 尾随垃圾。
2. 它**不**消除的变异：成员 mode（文件 0600 vs 0644、目录 0700 vs 0755）。
   这不是 bug 而是已知限制：跨 umask 的两次构建归一化后字节仍可能不同。
   若未来要修（把 mode 也钉死），本测试会故意失败 —— 那是提醒你必须重新基线
   dev10/dev11 的发布身份，而不是悄悄改掉。

用法：python tools/test_normalize_limits.py （0=PASS；缺官方 dev10 包时跳过 canary）
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from normalize_sdist import normalize  # noqa: E402

OFFICIAL = os.environ.get(
    "BNB_DEV10_SDIST",
    r"D:\work\_audit_release\gh_download\bitsandbytes_cpu_fork-0.50.2.dev10.tar.gz",
)
EXPECTED_DEV10_NORM = "b85cf8086d36fb948795e97ee7b2a95341860c15f64980721ef9c1c3d3eeef4a"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build(path, entries, fmt=tarfile.GNU_FORMAT):
    with tarfile.open(path, "w:gz", format=fmt) as t:
        for e in entries:
            ti = tarfile.TarInfo(e["name"])
            ti.mode = e.get("mode", 0o644)
            ti.mtime = e.get("mtime", 0)
            ti.uid = e.get("uid", 0)
            ti.gid = e.get("gid", 0)
            ti.uname = e.get("uname", "")
            ti.gname = e.get("gname", "")
            if e.get("dir"):
                ti.type = tarfile.DIRTYPE
                t.addfile(ti)
            elif e.get("link"):
                ti.type = tarfile.SYMTYPE
                ti.linkname = e["link"]
                t.addfile(ti)
            else:
                data = e.get("data", b"")
                ti.size = len(data)
                if e.get("pax"):
                    ti.pax_headers = e["pax"]
                t.addfile(ti, io.BytesIO(data))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="norm_limits_")
    failures = []

    def normalized(tag, entries, fmt=tarfile.GNU_FORMAT):
        src = os.path.join(tmp, tag + ".tar.gz")
        dst = os.path.join(tmp, tag + ".norm.tar.gz")
        build(src, entries, fmt=fmt)
        normalize(src, dst, 0)
        return sha(dst)

    def expect(name, got, want):
        ok = got == want
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        if not ok:
            failures.append(name)

    try:
        base = [
            {"name": "pkg/README.md", "data": b"hello"},
            {"name": "pkg/bitsandbytes/__init__.py", "data": b"x=1"},
            {"name": "pkg/", "dir": True, "mode": 0o755},
        ]
        ref = normalized("base", base)
        expect("baseline normalize", True, True)

        print("== 必须消除的变异 ==")
        expect(
            "mtime collapsed",
            normalized("mtime", [{**e, "mtime": 1234567890} for e in base]) == ref,
            True,
        )
        expect(
            "uid/gid collapsed",
            normalized("uid", [{**e, "uid": 1000, "gid": 1000} for e in base]) == ref,
            True,
        )
        expect(
            "uname/gname collapsed",
            normalized(
                "uname", [{**e, "uname": "alice", "gname": "staff"} for e in base]
            )
            == ref,
            True,
        )
        expect(
            "member order collapsed",
            normalized("rev", list(reversed(base))) == ref,
            True,
        )
        expect(
            "pax headers dropped",
            normalized(
                "pax",
                [{**e, "pax": {"comment": "build-xyz"}} for e in base],
                fmt=tarfile.PAX_FORMAT,
            )
            == ref,
            True,
        )

        print("== 已知限制（mode 未归一化）==")
        mode_file = normalized("mode_file", [{**e, "mode": 0o600} for e in base])
        expect(
            "file mode survives (known limitation; fixing it requires re-baselining)",
            mode_file != ref,
            True,
        )
        mode_dir = normalized(
            "mode_dir",
            [dict(e, mode=0o700) if e.get("dir") else dict(e) for e in base],
        )
        expect(
            "dir mode survives (known limitation; fixing it requires re-baselining)",
            mode_dir != ref,
            True,
        )

        print("== 幂等/链接/重复/垃圾尾 ==")
        once = os.path.join(tmp, "base.norm.tar.gz")
        twice = os.path.join(tmp, "base.norm2.tar.gz")
        normalize(once, twice, 0)
        expect("idempotent", sha(twice) == ref, True)

        links = base + [{"name": "pkg/link", "link": "README.md"}]
        expect(
            "symlink preserved deterministically",
            normalized("links", links) == normalized("links2", links),
            True,
        )

        dup = base + [{"name": "pkg/README.md", "data": b"hello"}]
        expect(
            "duplicate members deterministic",
            normalized("dup", dup) == normalized("dup2", dup),
            True,
        )

        garb = os.path.join(tmp, "garb.tar.gz")
        with open(garb, "wb") as f:
            f.write(open(os.path.join(tmp, "base.tar.gz"), "rb").read() + b"GARBAGE")
        norm_g = os.path.join(tmp, "garb.norm.tar.gz")
        normalize(garb, norm_g, 0)
        expect("trailing gzip garbage ignored", sha(norm_g) == ref, True)

        print("== 官方 dev10 canary ==")
        if os.path.isfile(OFFICIAL):
            out = os.path.join(tmp, "official.norm.tar.gz")
            normalize(OFFICIAL, out, 0)
            got = sha(out)
            expect("dev10 normalized sha pinned", got == EXPECTED_DEV10_NORM, True)
            if got != EXPECTED_DEV10_NORM:
                print("      got  %s" % got)
                print("      want %s" % EXPECTED_DEV10_NORM)
        else:
            print("  [SKIP] 官方 dev10 sdist 不存在: %s" % OFFICIAL)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("FAIL: %d check(s): %s" % (len(failures), failures))
        return 1
    print("PASS: normalize limits pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
