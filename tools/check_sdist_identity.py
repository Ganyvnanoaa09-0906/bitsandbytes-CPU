# -*- coding: utf-8 -*-
"""check_sdist_identity.py — 断言 sdist 内嵌 DLL 的**身份 + 形状**，并记录出处。

两种模式：

* **构建模式**（默认）：从干净 `git archive HEAD` 构建 sdist，注入审计过的 kernel DLL，
  再校验身份。它验证的是"干净树 + 注入 kernel"能产出正确身份，**不产生发布物**。
  注意：干净树里没有未跟踪的 `vcomp140.dll`，所以这条路径产出的 sdist 只有 1 个 DLL，
  **不是发布物形状**（dev10 发布的 sdist 是 2 个 DLL：kernel + vcomp140）。
* **审计模式**（`--sdist PATH`）：直接审计**已构建、将要上传的 sdist 字节**，不重建、不归一化。
  这是发布记录应使用的模式（规则：必须审计上传的字节，重建的产物即使内容相同也是另一个产物）。

形状策略（2026-11-16 修，依据 dev10 实测）：

* `libbitsandbytes_cpu.dll`：必须恰好 1 个，sha256 必须等于 `--expect-dll`
* `vcomp140.dll`（OpenMP 运行库）：
  - `--expect-vcomp-dll` 给出 ⇒ 必须恰好 1 个且 sha256 相等
  - 未给出但 sdist 里存在 ⇒ FAIL（必须显式决定是否随包发布，不允许静默）
  - `--sdist` 模式下未给出且不存在 ⇒ FAIL（dev10 发布物含它；除非有证据证明运行时不需要）
  - 构建模式下允许缺失（干净树本来就没有），但会打印明确警告
* 其它 `.dll` 一律 FAIL；`*.pyc/*.pyo/*.obj/...` 与 `__pycache__` 一律 FAIL

退出码：`0` 通过 / `1` 身份或形状失败 / `2` 出处未声明（或必填参数缺失）。

语义限度：`dll_build{source_commit, defines, toolchain}` 只是**操作者声明（attested）**，
不是机器证明；`provenance_verified` 恒为 False（证明需要同 commit+defines 重建并比对 `.text`）。

用法：
  # 审计将上传的 sdist（推荐）
  python tools/check_sdist_identity.py --sdist dist/pkg.tar.gz \
      --expect-dll <kernel sha256> --expect-vcomp-dll <vcomp sha256> \
      --dll-source-commit <commit> --dll-defines "" --dll-toolchain "<toolchain>"

  # 干净树构建演练（不是发布物形状）
  python tools/check_sdist_identity.py --dll-source <wheel|dll> --expect-dll <sha256> ...

  # 反向自测（无需构建，秒级）
  python tools/check_sdist_identity.py --selftest
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from normalize_sdist import normalize as _normalize_sdist
except Exception:  # 归一化是可选能力，缺失不应让身份门禁失效
    _normalize_sdist = None

FORBIDDEN = (
    ".pyc",
    ".pyo",
    ".obj",
    ".o",
    ".a",
    ".lib",
    ".exp",
    ".pdb",
    ".so",
    ".dylib",
)
KERNEL_DLL = "libbitsandbytes_cpu.dll"
VCOMP_DLL = "vcomp140.dll"
# 合法位置：<top>/bitsandbytes/<dll>（setup.py 的 package_data 布局）。只按 basename 匹配
# 会把任意目录下的同名 DLL 当成发布物 —— 那样审计结论不可信（对抗测试实测可绕过）。
DLL_DIR = "bitsandbytes"
# 资源上限：畸形/恶意 sdist 不得让审计器吃光内存。官方 dev10：DLL ~400 KB、成员 189。
# 两个上限都可被测试用环境变量调小。
MAX_DLL_BYTES = int(os.environ.get("BNB_SDIST_MAX_DLL_MB", "64")) << 20
MAX_MEMBERS = int(os.environ.get("BNB_SDIST_MAX_MEMBERS", "200000"))
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _fatal(msg):
    """用法/环境错误：exit 2（1 保留给"审计结论为失败"）。"""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _at_package_root(member):
    """sdist 成员是否位于 <top>/bitsandbytes/ 下（深度恰好 3）。"""
    parts = member.name.split("/")
    return len(parts) == 3 and parts[1] == DLL_DIR


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def classify_sdist(path, expect_kernel, expect_vcomp, require_vcomp):
    """审计一个 sdist 的字节，返回身份/形状结果（纯函数，不做任何构建）。

    require_vcomp=True 表示"发布物必须带 vcomp140.dll"（--sdist 模式的策略）。
    形状规则（对抗测试驱动，全部 fail closed）：
      * DLL 必须位于 <top>/bitsandbytes/<name>，错误位置不接受（哪怕 basename 对、sha 对）
      * kernel 恰好 1 个、vcomp 策略见参数；其它 DLL 一律拒绝
      * 路径穿越 / 绝对路径成员拒绝；构建产物/字节码成员拒绝（大小写不敏感）
      * 单个 DLL 超过 MAX_DLL_BYTES、成员总数超过 MAX_MEMBERS 拒绝
    """
    with tarfile.open(path) as t:
        members = t.getmembers()
        too_many = len(members) > MAX_MEMBERS
        files = [m for m in members if m.isfile()]
        names = [m.name for m in files]
        all_names = [m.name for m in members]
        unsafe = [n for n in all_names if n.startswith("/") or ".." in n.split("/")]
        dlls = [m for m in files if m.name.lower().endswith(".dll")]
        kname, vname = KERNEL_DLL.lower(), VCOMP_DLL.lower()
        kernels = [
            m
            for m in dlls
            if _at_package_root(m) and m.name.split("/")[2].lower() == kname
        ]
        vcomps = [
            m
            for m in dlls
            if _at_package_root(m) and m.name.split("/")[2].lower() == vname
        ]
        misplaced_kernels = [
            m
            for m in dlls
            if m.name.split("/")[-1].lower() == kname and m not in kernels
        ]
        others = [m for m in dlls if m not in kernels and m not in vcomps]
        # 大小写不敏感：Windows 上 .PYC / __PyCache__ 与 linux 上的小写是同一类污染
        bad = [
            n
            for n in names
            if n.lower().endswith(FORBIDDEN) or "__pycache__" in n.lower()
        ]
        oversized = [m.name for m in dlls if m.size > MAX_DLL_BYTES]
        kernel_sha = None
        vcomp_sha = None
        if not too_many and not oversized:
            if len(kernels) == 1:
                kernel_sha = sha_bytes(t.extractfile(kernels[0]).read())
            if len(vcomps) == 1:
                vcomp_sha = sha_bytes(t.extractfile(vcomps[0]).read())
        member_count = len(files)
        member_total = len(members)

    errors = []
    if too_many:
        errors.append(
            "成员总数=%d 超过上限 %d（畸形 sdist？）" % (member_total, MAX_MEMBERS)
        )
    if unsafe:
        errors.append("路径穿越/绝对路径成员: %s" % unsafe[:5])
    if oversized:
        errors.append(
            "DLL 成员超过 %d MiB 上限: %s" % (MAX_DLL_BYTES >> 20, oversized[:3])
        )
    if len(kernels) != 1:
        msg = "kernel DLL 数量=%d（必须恰好 1 个，且位于 <top>/%s/）" % (
            len(kernels),
            DLL_DIR,
        )
        if misplaced_kernels:
            msg += "；另有同名 DLL 在错误位置: %s" % [
                m.name for m in misplaced_kernels[:3]
            ]
        errors.append(msg)
    elif kernel_sha != expect_kernel:
        errors.append(
            "kernel DLL 身份不符: %s != %s"
            % ((kernel_sha or "?")[:16], expect_kernel[:16])
        )

    if expect_vcomp:
        if len(vcomps) != 1:
            errors.append(
                "vcomp140.dll 数量=%d（要求恰好 1 个，且位于 <top>/%s/）"
                % (len(vcomps), DLL_DIR)
            )
        elif vcomp_sha != expect_vcomp:
            errors.append(
                "vcomp140.dll 身份不符: %s != %s"
                % ((vcomp_sha or "?")[:16], expect_vcomp[:16])
            )
    else:
        if vcomps:
            errors.append(
                "sdist 含 vcomp140.dll 但未给 --expect-vcomp-dll："
                "必须显式决定它是否随包发布（给期望值，或从不含它的树构建）"
            )
        elif require_vcomp:
            errors.append(
                "sdist 缺 vcomp140.dll：dev10 发布物包含它；"
                "除非有证据证明运行时不需要，否则发布物必须包含"
            )

    if others:
        errors.append(
            "出现未预期的 DLL: %s" % [os.path.basename(m.name) for m in others]
        )
    if bad:
        errors.append("构建产物/字节码成员: %s" % bad[:5])

    identity_ok = len(kernels) == 1 and kernel_sha == expect_kernel
    shape_ok = not errors
    return {
        "sdist": path,
        "sdist_bytes": os.path.getsize(path),
        "sdist_sha256": sha_file(path),
        "member_count": member_count,
        "member_total": member_total,
        "kernel_dll_sha256": kernel_sha,
        "vcomp_dll_sha256": vcomp_sha,
        "dll_count": len(dlls),
        "dll_names": [os.path.basename(m.name) for m in dlls],
        "extra_dlls": [os.path.basename(m.name) for m in others],
        "misplaced_kernels": [m.name for m in misplaced_kernels],
        "unsafe_members": unsafe,
        "forbidden_members": bad,
        "identity_ok": identity_ok,
        "shape_ok": shape_ok,
        "errors": errors,
    }


def _write_synthetic_sdist(path, files, links=()):
    """files: list of (name, bytes)；links: list of (name, target)。用于自测，不依赖构建工具链。"""
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


def _selftest(tmp):
    """反向自测：门禁必须能对每一种坏形状明确失败。返回 0=全过 / 1=有断言失败。"""
    k = b"KERNEL-BYTES-" + b"k" * 100
    v = b"VCOMP-BYTES-" + b"v" * 100
    ksha, vsha = sha_bytes(k), sha_bytes(v)
    pkg = "pkg-1.0/bitsandbytes/"
    cases = [
        # (label, files, expect_kernel, expect_vcomp, require_vcomp, should_be_clean)
        (
            "kernel+vcomp 正确",
            [(pkg + KERNEL_DLL, k), (pkg + VCOMP_DLL, v)],
            ksha,
            vsha,
            True,
            True,
        ),
        (
            "kernel 身份错误",
            [(pkg + KERNEL_DLL, b"STALE" + k), (pkg + VCOMP_DLL, v)],
            ksha,
            vsha,
            True,
            False,
        ),
        ("缺 vcomp（发布策略）", [(pkg + KERNEL_DLL, k)], ksha, vsha, True, False),
        ("缺 vcomp（构建模式允许）", [(pkg + KERNEL_DLL, k)], ksha, None, False, True),
        # LEAD 独立复核时踩到的真实形态：DLL 放错目录（放到 setup.py 旁而不是内层包目录）时
        # setup.py sdist 仍 rc=0，但产物 187 成员 / 0 DLL —— 静默丢 DLL，构建"成功"。
        # 门禁必须对这种形态 fail closed。
        (
            "kernel 缺失（静默丢 DLL）",
            [(pkg + VCOMP_DLL, v)],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "完全无 DLL（setup.py rc=0 的静默失败）",
            [(pkg + "some_module.py", b"x")],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "vcomp 身份错误",
            [(pkg + KERNEL_DLL, k), (pkg + VCOMP_DLL, b"WRONG" + v)],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "多余 DLL",
            [(pkg + KERNEL_DLL, k), (pkg + VCOMP_DLL, v), (pkg + "extra.dll", b"x")],
            ksha,
            vsha,
            True,
            False,
        ),
        # LEAD 复核补充：同一尺寸下可能有两个不同 kernel（403,968 B：dev9 stale `3be2a7fe…`
        # vs 构建产物 `1bbac05f…`；官方 dev10 才是 402,944 B `472675df…`）。
        # 身份只能按 sha256，size/形状不参与判定 —— 本用例保证"同尺寸不同内容"必须 FAIL。
        (
            "kernel 同尺寸不同 sha（size 不参与身份）",
            [(pkg + KERNEL_DLL, b"X" + k[1:]), (pkg + VCOMP_DLL, v)],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "pyc 成员",
            [(pkg + KERNEL_DLL, k), (pkg + VCOMP_DLL, v), (pkg + "a.pyc", b"x")],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "有 vcomp 但未声明期望值",
            [(pkg + KERNEL_DLL, k), (pkg + VCOMP_DLL, v)],
            ksha,
            None,
            False,
            False,
        ),
        # 以下为 2026-11-16 对抗测试新增：这些形状此前会被放行（审计结论不可信）。
        (
            "kernel 在错误目录（pkg/other/）",
            [("pkg-1.0/other/" + KERNEL_DLL, k), (pkg + VCOMP_DLL, v)],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "vcomp 在错误目录",
            [(pkg + KERNEL_DLL, k), ("pkg-1.0/other/" + VCOMP_DLL, v)],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "路径穿越成员 ../../evil",
            [("../../evil", b"x"), (pkg + KERNEL_DLL, k), (pkg + VCOMP_DLL, v)],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "大写 .PYC 成员",
            [(pkg + KERNEL_DLL, k), (pkg + VCOMP_DLL, v), (pkg + "a.PYC", b"x")],
            ksha,
            vsha,
            True,
            False,
        ),
        (
            "kernel 是符号链接",
            [(pkg + VCOMP_DLL, v)],
            ksha,
            vsha,
            True,
            False,
            [(pkg + KERNEL_DLL, "/etc/passwd")],
        ),
    ]
    failed = 0
    for i, case in enumerate(cases, 1):
        label, files, ek, ev, rv, want_clean = case[:6]
        links = case[6] if len(case) > 6 else ()
        p = os.path.join(tmp, "case%d.tar.gz" % i)
        _write_synthetic_sdist(p, files, links)
        r = classify_sdist(p, ek, ev, rv)
        got_clean = r["shape_ok"]
        ok = got_clean == want_clean
        print(
            "  [%s] case%d %-28s want=%s got=%s%s"
            % (
                "PASS" if ok else "FAIL",
                i,
                label,
                want_clean,
                got_clean,
                "" if got_clean else "  errors=%s" % r["errors"],
            )
        )
        if not ok:
            failed += 1
    print(
        "自测: %s（%d/%d）"
        % ("PASS" if failed == 0 else "FAIL", len(cases) - failed, len(cases))
    )
    return 0 if failed == 0 else 1


def repo_root(explicit):
    if explicit:
        return os.path.abspath(explicit)
    r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if r.returncode != 0:
        _fatal("[FATAL] 不是 git 仓库且未给 --root: %s" % r.stderr.strip())
    return r.stdout.strip()


def head_commit(root):
    return subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def _optional_tree_head(root_arg):
    """审计模式不要求 git 仓库（发布机/CI 可能只有下载下来的 sdist）；不在仓库内返回 None。

    这里不能用 repo_root()：它失败时会打印 [FATAL]（那是给"必须找到仓库"的构建模式用的），
    而审计模式把"没有仓库"当作正常情况，所以静默探测。
    """
    if root_arg:
        try:
            return head_commit(os.path.abspath(root_arg))
        except OSError:
            return None
    r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    try:
        return head_commit(r.stdout.strip())
    except OSError:
        return None


def export_clean(root, dest):
    tar = os.path.join(dest, "tree.tar")
    subprocess.run(
        ["git", "-C", root, "archive", "-o", tar, "--format=tar", "HEAD"], check=True
    )
    tree = os.path.join(dest, "tree")
    os.makedirs(tree, exist_ok=True)
    subprocess.run(["tar", "-xf", tar, "-C", tree], check=True)
    os.remove(tar)
    return tree


def dll_blob(source):
    if source.lower().endswith(".whl"):
        with zipfile.ZipFile(source) as z:
            names = [n for n in z.namelist() if n.endswith(KERNEL_DLL)]
            if len(names) != 1:
                _fatal("[FATAL] wheel 里 kernel DLL 数量=%d" % len(names))
            return z.read(names[0]), "%s::%s" % (os.path.basename(source), names[0])
    if not os.path.isfile(source):
        _fatal("[FATAL] DLL 来源不存在: %s" % source)
    return io.open(source, "rb").read(), os.path.abspath(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="跑反向自测（合成 sdist，不构建、不联网），exit 0=PASS",
    )
    ap.add_argument(
        "--sdist",
        default=None,
        help="审计一个已构建的 sdist（不重建、不归一化）；发布记录应使用本模式",
    )
    ap.add_argument(
        "--expect-dll",
        default=None,
        help="本次发布审计过的 kernel DLL sha256（显式给出，无默认值）",
    )
    ap.add_argument(
        "--expect-vcomp-dll",
        default=None,
        help="vcomp140.dll 的期望 sha256；发布物含它时必须给",
    )
    ap.add_argument(
        "--dll-source", default=None, help="构建模式：wheel 或单个 DLL 文件"
    )
    ap.add_argument(
        "--dll-source-commit", default=None, help="构建该 DLL 的源码 commit"
    )
    ap.add_argument(
        "--dll-defines", default=None, help="构建定义（-D 列表），无则传空串"
    )
    ap.add_argument(
        "--dll-toolchain",
        default=None,
        help='工具链描述，例如 "zig 0.14.1 windows-gnu"',
    )
    ap.add_argument("--root", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="（构建模式默认开启）构建后做确定性归一化；用 --no-normalize 关闭",
    )
    ap.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="关闭归一化；此时发布物身份就是原始 sha256",
    )
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        tmp = tempfile.mkdtemp(prefix="sdist_identity_selftest_")
        try:
            return _selftest(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if a.sdist and a.dll_source:
        _fatal("[FATAL] --sdist 与 --dll-source 互斥（审计上传字节时不重建）")
    if not a.sdist and not a.dll_source:
        _fatal("[FATAL] 需要 --sdist（审计）或 --dll-source（构建演练）")
    if not a.expect_dll:
        _fatal("[FATAL] 缺少 --expect-dll（本次发布的 kernel DLL sha256，无默认值）")
    if not _HEX64.fullmatch(a.expect_dll):
        _fatal("[FATAL] --expect-dll 必须是 64 位小写十六进制 sha256")
    if a.expect_vcomp_dll and not _HEX64.fullmatch(a.expect_vcomp_dll):
        _fatal("[FATAL] --expect-vcomp-dll 必须是 64 位小写十六进制 sha256")

    provenance_attested = (
        bool(a.dll_source_commit)
        and a.dll_defines is not None
        and bool(a.dll_toolchain)
    )
    # 真正的证明需要"用该 commit + 该 defines 重建一次并比对 .text"，本工具不做重建。
    provenance_verified = False

    # ---------------- 审计模式：不重建，直接审上传字节 ----------------
    if a.sdist:
        if not os.path.isfile(a.sdist):
            _fatal("[FATAL] sdist 不存在: %s" % a.sdist)
        try:
            r = classify_sdist(
                a.sdist,
                a.expect_dll,
                a.expect_vcomp_dll,
                require_vcomp=bool(a.expect_vcomp_dll),
            )
        except (tarfile.TarError, EOFError, OSError) as e:
            print("审计 sdist: %s" % a.sdist)
            print(
                "  [FAIL] 无法解析 sdist（不是有效的 tar.gz？）: %s: %s"
                % (type(e).__name__, e)
            )
            print("结果: FAIL（无法解析，未做任何身份判定）")
            return 1
        print("审计 sdist: %s" % a.sdist)
        print(
            "  字节数 %d  成员 %d  sha256=%s"
            % (r["sdist_bytes"], r["member_count"], r["sdist_sha256"][:16])
        )
        print("  DLL 数 %d: %s" % (r["dll_count"], r["dll_names"]))
        print(
            "  kernel sha256 = %s  %s"
            % (
                (r["kernel_dll_sha256"] or "?")[:16],
                "OK" if r["identity_ok"] else "MISMATCH",
            )
        )
        print(
            "  vcomp  sha256 = %s  %s"
            % (
                (r["vcomp_dll_sha256"] or "?")[:16],
                "OK"
                if (
                    not a.expect_vcomp_dll
                    or r["vcomp_dll_sha256"] == a.expect_vcomp_dll
                )
                else "MISMATCH",
            )
        )
        print(
            "  出处已声明(attested) = %s (commit=%s toolchain=%s)；已验证(verified) = %s"
            % (
                provenance_attested,
                (a.dll_source_commit or "?")[:12],
                a.dll_toolchain or "?",
                provenance_verified,
            )
        )
        for e in r["errors"]:
            print("  [FAIL] %s" % e)
        rec = {
            "mode": "audit-uploaded-bytes",
            "tree_head": _optional_tree_head(a.root),
            "audited_artifact": os.path.abspath(a.sdist),
            "audited_artifact_sha256": r["sdist_sha256"],
            "publish_sha256": r["sdist_sha256"],
            "publish_basis": "uploaded-bytes",
            "dll_build": {
                "source_commit": a.dll_source_commit,
                "defines": a.dll_defines,
                "toolchain": a.dll_toolchain,
            },
            "provenance_attested": provenance_attested,
            "provenance_verified": provenance_verified,
            "provenance_note": (
                "attested = operator declares commit/defines/toolchain; "
                "verified = would require rebuilding from that commit and "
                "comparing .text, which this tool does not do"
            ),
            "sdist_bytes": r["sdist_bytes"],
            "member_count": r["member_count"],
            "dll_count": r["dll_count"],
            "dll_names": r["dll_names"],
            "kernel_dll_sha256": r["kernel_dll_sha256"],
            "vcomp_dll_sha256": r["vcomp_dll_sha256"],
            "expect_kernel_dll_sha256": a.expect_dll,
            "expect_vcomp_dll_sha256": a.expect_vcomp_dll,
            "identity_ok": r["identity_ok"],
            "shape_ok": r["shape_ok"],
            "errors": r["errors"],
        }
        if a.json:
            try:
                io.open(a.json, "w", encoding="utf-8", newline="").write(
                    json.dumps(rec, ensure_ascii=False, indent=2)
                )
            except OSError as e:
                print("  [FAIL] 无法写 --json 记录: %s" % e)
                print("结果: FAIL（记录写不出来，审计不算完成）")
                return 1
            print("记录: %s" % a.json)
        if not (r["identity_ok"] and r["shape_ok"]):
            print("结果: FAIL（身份或形状）")
            return 1
        if not provenance_attested:
            print(
                "结果: FAIL（出处未声明：--dll-source-commit / --dll-defines / --dll-toolchain）"
            )
            return 2
        print("结果: PASS（身份 + 形状 + 出处；审计的是上传字节）")
        return 0

    # ---------------- 构建模式（干净树演练；不是发布物形状）----------------
    root = repo_root(a.root)
    tmp = tempfile.mkdtemp(prefix="sdist_identity_")
    try:
        tree = export_clean(root, tmp)
        pkg = os.path.join(tree, "bitsandbytes")
        if not os.path.isdir(pkg):
            _fatal("[FATAL] git archive 里没有 bitsandbytes/")

        target = os.path.join(pkg, "bitsandbytes", KERNEL_DLL)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        blob, src_desc = dll_blob(a.dll_source)
        io.open(target, "wb").write(blob)
        got = sha_bytes(blob)
        print("注入 DLL: %s" % src_desc)
        print("  来源 sha256 = %s  (%d B)" % (got[:16], len(blob)))
        if got != a.expect_dll:
            _fatal("[FATAL] 注入源本身就不是审计值: %s != %s" % (got, a.expect_dll))
        if a.expect_vcomp_dll:
            print(
                "  [warn] 构建模式基于 git archive，不含未跟踪的 vcomp140.dll；"
                "本产物不是发布物形状，发布审计请用 --sdist"
            )

        for d in os.listdir(pkg):
            if d.endswith(".egg-info"):
                shutil.rmtree(os.path.join(pkg, d))

        outdir = os.path.join(tmp, "dist")
        subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--outdir", outdir],
            cwd=pkg,
            check=True,
            capture_output=True,
        )
        names = [f for f in os.listdir(outdir) if f.endswith(".tar.gz")]
        if len(names) != 1:
            _fatal("[FATAL] sdist 数量=%d" % len(names))
        sdist = os.path.join(outdir, names[0])

        r = classify_sdist(sdist, a.expect_dll, a.expect_vcomp_dll, require_vcomp=False)
        print(
            "sdist: %d B  成员 %d  sha256=%s"
            % (r["sdist_bytes"], r["member_count"], r["sdist_sha256"][:16])
        )
        print("  DLL 数 %d: %s" % (r["dll_count"], r["dll_names"]))
        print(
            "  kernel sha256 = %s  %s"
            % (
                (r["kernel_dll_sha256"] or "?")[:16],
                "OK" if r["identity_ok"] else "MISMATCH",
            )
        )
        print("  构建产物/字节码违规 = %d" % len(r["forbidden_members"]))
        print(
            "  出处已声明(attested) = %s (commit=%s toolchain=%s)；已验证(verified) = %s"
            % (
                provenance_attested,
                (a.dll_source_commit or "?")[:12],
                a.dll_toolchain or "?",
                provenance_verified,
            )
        )
        for e in r["errors"]:
            print("  [FAIL] %s" % e)

        norm_path = None
        norm_sha = None
        norm_bytes = None
        if a.normalize:
            if _normalize_sdist is None:
                _fatal("[FATAL] --normalize 需要 tools/normalize_sdist.py")
            norm_path = os.path.join(
                outdir, names[0].replace(".tar.gz", ".normalized.tar.gz")
            )
            _normalize_sdist(sdist, norm_path, 0)
            norm_sha = sha_file(norm_path)
            norm_bytes = os.path.getsize(norm_path)
            nr = classify_sdist(
                norm_path, a.expect_dll, a.expect_vcomp_dll, require_vcomp=False
            )
            print(
                "  归一化后: %d B  sha256=%s  内嵌 kernel=%s"
                % (norm_bytes, norm_sha[:16], "OK" if nr["identity_ok"] else "MISMATCH")
            )
            if not nr["identity_ok"]:
                _fatal("[FATAL] 归一化改变了内嵌 kernel DLL")

        rec = {
            "mode": "clean-tree-rehearsal",
            "tree_head": head_commit(root),
            "built_from": "git archive HEAD",
            "dll_build": {
                "source_commit": a.dll_source_commit,
                "defines": a.dll_defines,
                "toolchain": a.dll_toolchain,
            },
            "provenance_attested": provenance_attested,
            "provenance_verified": provenance_verified,
            "provenance_note": (
                "attested = operator declares commit/defines/toolchain; "
                "verified = would require rebuilding from that commit and "
                "comparing .text, which this tool does not do"
            ),
            "dll_source": src_desc,
            "dll_source_sha256": got,
            "expect_dll_sha256": a.expect_dll,
            "embedded_dll_sha256": r["kernel_dll_sha256"],
            "identity_ok": r["identity_ok"],
            "shape_ok": r["shape_ok"],
            "shape_warning": (
                "clean-tree build omits untracked vcomp140.dll; "
                "not the release-artifact shape"
            ),
            "forbidden_members": r["forbidden_members"],
            "sdist_name": names[0],
            "sdist_bytes": r["sdist_bytes"],
            "sdist_sha256": r["sdist_sha256"],
            "sdist_sha256_normalized": norm_sha,
            "publish_sha256": norm_sha if a.normalize else r["sdist_sha256"],
            "publish_basis": "normalized" if a.normalize else "raw",
            "sdist_normalized_bytes": norm_bytes,
            "normalized": bool(a.normalize),
            "member_count": r["member_count"],
        }
        basis = "normalized" if a.normalize else "raw"
        pub = norm_sha if a.normalize else r["sdist_sha256"]
        print("  发布物身份(%s) = %s" % (basis, pub[:16]))
        if a.json:
            io.open(a.json, "w", encoding="utf-8", newline="").write(
                json.dumps(rec, ensure_ascii=False, indent=2)
            )
            print("记录: %s" % a.json)

        if not (r["identity_ok"] and r["shape_ok"]):
            print("结果: FAIL（身份或形状）")
            return 1
        if not provenance_attested:
            print(
                "结果: FAIL（出处未声明：--dll-source-commit / --dll-defines / --dll-toolchain）"
            )
            return 2
        print(
            "结果: PASS（构建演练：身份 + 形状 + 出处；产物不是发布物形状，发布审计用 --sdist）"
        )
        return 0
    finally:
        if a.keep:
            print("[keep] %s" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
