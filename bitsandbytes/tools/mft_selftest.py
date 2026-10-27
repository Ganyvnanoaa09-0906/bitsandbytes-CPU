#!/usr/bin/env python3
"""mft_selftest.py — mft_recover 的自动化验收（真 NTFS 卷，不是模拟数据）。

思路：
  1. 用 WSL 里的 mkfs.ntfs 造一个真 NTFS 卷（可指定簇大小等几何参数），
     用 ntfs-3g 挂上，写一批内容随机的文件，**删前记录每个文件的 sha256**，然后 rm 掉要恢复的那批
  2. 用 mft_recover 从镜像里恢复
  3. 逐个比对 sha256 —— 恢复出来的字节必须和删除前完全一致，不是「大小差不多」

覆盖场景：
  basic    默认几何（512B 扇区 / 4KB 簇 / 1KB 记录 = 每簇 4 条记录）
  geom8k   8KB 簇（每簇 8 条记录；曾经「只读簇首」的 bug 会漏掉 7/8）
  geom1k   1KB 簇（每簇 1 条记录，记录大小==簇大小的边界）
  s4k      4KB 扇区（fixup 的 USA 按 4KB 边界替换的另一种布局）
  damaged  人为把某个已删除文件的 MFT 记录整条清零（模拟坏道），要求不崩、其余文件照样还原

用法：
  python mft_selftest.py                      # 全部场景
  python mft_selftest.py --only basic         # 只跑一个
  python mft_selftest.py --workdir D:\\tmp\\mft --keep    # 保留现场
环境：
  需要 WSL（默认发行版）+ ntfs-3g；没有则相关场景标记 SKIP 而不是 FAIL。
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXE = os.path.join(HERE, "mft_recover.exe")

BUILDER = r"""#!/bin/bash
# 由 mft_selftest.py 生成：造一个真 NTFS 卷，写文件，记录 sha256，删掉待恢复的那批
set -u
IMG="$1"; MNT="$2"; MAN="$3"; shift 3
GEO="$*"

rm -f "$IMG" "$MAN"
truncate -s 24M "$IMG" || { echo "TRUNCATE_FAIL"; exit 2; }
# shellcheck disable=SC2086
mkfs.ntfs -F -Q -L mfttest $GEO "$IMG" >/dev/null 2>&1 || { echo "MKFS_FAIL"; exit 2; }
mkdir -p "$MNT"
LOOP=$(losetup -f --show "$IMG" 2>/dev/null) || { echo "LOSETUP_FAIL"; exit 2; }
mount -t ntfs-3g "$LOOP" "$MNT" 2>/dev/null || { echo "MOUNT_FAIL"; losetup -d "$LOOP" 2>/dev/null; exit 2; }

cd "$MNT" || { echo "CD_FAIL"; exit 2; }
head -c 17     /dev/urandom > small.txt
head -c 204800 /dev/urandom > big.bin
: > zero.dat
mkdir -p nested
head -c 40     /dev/urandom > nested/deep.txt
head -c 30     /dev/urandom > china_name.txt
head -c 64     /dev/urandom > live1.txt
head -c 5000   /dev/urandom > live2.bin
sync

add() {
  st="$1"; shift
  for f in "$@"; do
    sz=$(stat -c %s "$f")
    hs=$(sha256sum "$f" | cut -d' ' -f1)
    printf '%s\t%s\t%s\t%s\n' "${f#./}" "$hs" "$sz" "$st" >> "$MAN"
  done
}
add live live1.txt live2.bin
add deleted small.txt big.bin zero.dat nested/deep.txt china_name.txt

rm -f small.txt big.bin zero.dat nested/deep.txt china_name.txt
rmdir nested 2>/dev/null
sync
cd / || true
umount "$MNT" 2>/dev/null || umount -l "$MNT" 2>/dev/null || { echo "UMOUNT_FAIL"; exit 2; }
losetup -d "$LOOP" 2>/dev/null
echo "BUILD_OK"
"""

# 每个场景：名字 -> mkfs.ntfs 额外参数
CASES = {
    "basic": [],
    "geom8k": ["-c", "8192"],
    "geom1k": ["-c", "1024"],
    "s4k": ["-s", "4096"],
}


def wsl_path(win_path):
    """D:\\a\\b -> /mnt/d/a/b（WSL 默认挂载点）"""
    p = os.path.abspath(win_path).replace("\\", "/")
    drive = p[0].lower()
    return "/mnt/%s%s" % (drive, p[2:])


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("errors", "replace")
    return subprocess.run(cmd, **kw)


def build_image(case_dir, geo):
    """调 WSL 造卷，返回 {relpath: (sha256, size, state)}；失败抛 RuntimeError"""
    os.makedirs(case_dir, exist_ok=True)
    img = os.path.join(case_dir, "mft.img")
    mnt = os.path.join(case_dir, "mnt")
    man = os.path.join(case_dir, "manifest.tsv")
    script = os.path.join(case_dir, "_build.sh")
    with open(script, "w", newline="\n", encoding="utf-8") as f:
        f.write(BUILDER)
    os.makedirs(mnt, exist_ok=True)
    r = run(["wsl", "-u", "root", "--", "bash", wsl_path(script),
             wsl_path(img), wsl_path(mnt), wsl_path(man)] + geo)
    if r.returncode != 0 or "BUILD_OK" not in r.stdout:
        raise RuntimeError("造卷失败: rc=%s out=%r err=%r" % (r.returncode, r.stdout[-300:], r.stderr[-300:]))
    expect = {}
    with open(man, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rel, hs, sz, st = line.split("\t")
            expect[rel] = (hs, int(sz), st)
    return img, expect


def parse_params(stdout):
    m = re.search(r"cluster=(\d+) MFT_LCN=(\d+) recSize=(\d+)", stdout)
    if not m:
        raise RuntimeError("拿不到卷参数: %r" % stdout[:300])
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def run_tool(exe, img, outdir, extra=()):
    if outdir and os.path.isdir(outdir):
        shutil.rmtree(outdir)
    args = [exe, img]
    if outdir:
        args.append(outdir)
    args.extend(extra)
    r = run(args)
    return r


def check_recovered(outdir, expect, states, label, allow_missing=()):
    """返回 (ok, lines)。用 sha256 做多重集比对，忽略文件名里的记录号前缀"""
    want = {}
    for rel, (hs, sz, st) in expect.items():
        if st in states:
            want[os.path.basename(rel)] = (hs, sz)
    allow = {os.path.basename(x) for x in allow_missing}
    got = []
    if os.path.isdir(outdir):
        for fn in sorted(os.listdir(outdir)):
            p = os.path.join(outdir, fn)
            if not os.path.isfile(p):
                continue
            m = re.match(r"^(del|live)_(\d{6})_(.+)$", fn)
            base = m.group(3) if m else fn
            got.append((base, sha256_file(p), os.path.getsize(p), fn))
    lines = []
    ok = True
    used = set()
    for base, (hs, sz) in sorted(want.items()):
        hit = None
        for i, (gbase, ghs, gsz, gfn) in enumerate(got):
            if i in used:
                continue
            if ghs == hs and gsz == sz:
                hit = i
                break
        if hit is None:
            if base in allow:
                lines.append("  SKIP %-18s (%d B) — 记录被故意损坏，允许缺失" % (base, sz))
            else:
                lines.append("  MISS %-18s (%d B) — 没恢复出来" % (base, sz))
                ok = False
        else:
            used.add(hit)
            gbase, ghs, gsz, gfn = got[hit]
            name_ok = (base in gbase) or (gbase in base)
            lines.append("  OK   %-18s (%d B) -> %s%s" % (base, sz, gfn, "" if name_ok else "  [名字不符]"))
            if not name_ok:
                ok = False
    extra = [g for i, g in enumerate(got) if i not in used]
    for gbase, ghs, gsz, gfn in extra:
        lines.append("  EXTRA %-17s (%d B) -> %s" % (gbase, gsz, gfn))
    # 恢复出多余文件不算失败（系统文件、$MFT 等），但名字完全不该出现才提示
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "mft_selftest"))
    ap.add_argument("--only", default=None, help="只跑指定场景（逗号分隔）")
    ap.add_argument("--keep", action="store_true", help="保留镜像与输出便于排查")
    args = ap.parse_args()

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print("找不到 mft_recover.exe: %s" % exe)
        print("先编译: cl /nologo /O2 /utf-8 mft_recover.c /link advapi32.lib")
        return 2

    cases = list(CASES)
    if args.only:
        cases = [c for c in args.only.split(",") if c]
    cases = cases + ["damaged"]

    results = {}
    for case in cases:
        print("\n" + "=" * 70)
        print("场景 %s" % case)
        print("=" * 70)
        case_dir = os.path.join(args.workdir, case)
        try:
            if case == "damaged":
                geo = CASES["basic"]
            else:
                geo = CASES[case]
            img, expect = build_image(case_dir, geo)
        except Exception as e:  # noqa: BLE001
            print("  SKIP 造卷不可用: %s" % e)
            results[case] = (None, ["  SKIP %s" % e])
            continue

        outdir = os.path.join(case_dir, "out")
        r = run_tool(exe, img, outdir)
        print("  tool rc=%d" % r.returncode)
        if r.returncode != 0:
            results[case] = (False, ["  tool 返回码非 0: %r" % r.stderr[-300:]])
            continue

        if case == "damaged":
            # 找到 big.bin 的记录号，把它整条记录清零，再恢复
            dbg = run_tool(exe, img, None, ["--debug"])
            mm = re.search(r"\[dbg\] rec (\d+) @0x[0-9a-f]+ in_use=\d+ dir=\d+ name=big\.bin",
                           dbg.stdout)
            lines = []
            if not mm:
                results[case] = (False, ["  找不到 big.bin 的 MFT 记录，无法构造坏道场景"])
                continue
            rec = int(mm.group(1))
            cluster, mft_lcn, rec_size = parse_params(dbg.stdout)
            off = mft_lcn * cluster + rec * rec_size
            with open(img, "r+b") as f:
                f.seek(off)
                f.write(b"\x00" * rec_size)
            print("  已把 rec %d (big.bin) @0x%x 整条清零" % (rec, off))
            r2 = run_tool(exe, img, outdir)
            print("  tool rc=%d" % r2.returncode)
            ok = r2.returncode == 0
            ok2, lines2 = check_recovered(outdir, expect, {"deleted"},
                                          case, allow_missing=("big.bin",))
            lines += lines2
            ok = ok and ok2
            results[case] = (ok, lines)
            for ln in lines:
                print(ln)
            continue

        ok, lines = check_recovered(outdir, expect, {"deleted"}, case)
        # --all 模式：未删除的文件也要能按名字还原
        outall = os.path.join(case_dir, "out_all")
        rall = run_tool(exe, img, outall, ["--all"])
        if rall.returncode != 0:
            ok = False
            lines.append("  FAIL --all 返回码 %d" % rall.returncode)
        else:
            ok_all, lines_all = check_recovered(outall, expect, {"live"}, case + "/all")
            lines.append("  --all:")
            lines.extend(lines_all)
            ok = ok and ok_all
        results[case] = (ok, lines)
        for ln in lines:
            print(ln)

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    fails = 0
    skips = 0
    for case in cases:
        ok, _ = results.get(case, (None, []))
        state = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        if state == "FAIL":
            fails += 1
        if state == "SKIP":
            skips += 1
        print("  %-10s %s" % (case, state))
    if not args.keep:
        shutil.rmtree(args.workdir, ignore_errors=True)
    print("\n结果: %d 失败, %d 跳过, 共 %d 场景" % (fails, skips, len(cases)))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
