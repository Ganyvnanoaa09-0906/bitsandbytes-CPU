#!/usr/bin/env python3
"""recovery_pipeline_selftest.py — 验证文档 §9.2.6 里那条「先判断、再恢复」的决策链。

QUICKSTART 写的是：
    MFT 还能读  ->  mft_recover   （有名字、大小、目录树，字节精确）
    MFT 已被毁  ->  sector_carve  （靠签名硬扫，没名字但能把内容捞回来）

这个脚本就在真 NTFS 卷上把两条路都跑一遍，证明文档没说错：

  1. WSL 造一个 64MB 真 NTFS 卷，写入 9 个真文件（2 真 JPEG + 2 PNG + GIF + ZIP + docx/xlsx/pptx），
     删掉它们（数据扇区仍在，MFT 记录被标记为已删除）
  2. 分支 A（MFT 活着）：mft_recover 恢复 -> 9 个文件必须逐字节一致
  3. 分支 B（MFT 被毁）：把 $MFT 自己的数据区整段清零（模拟 MFT 被覆盖），
     - mft_recover 必须「不崩、报 0 个导出」（它只做元数据恢复，元数据没了就只能说没有）
     - sector_carve 必须按签名把内容捞回来，逐字节一致
  4. 两个分支都过 -> PASS

用法：
  python recovery_pipeline_selftest.py [--keep] [--workdir D:\\work\\_pipeline] [--exe-dir DIR]
环境：需要 WSL（默认发行版）+ ntfs-3g；需要 Pillow 生成 PNG/GIF（JPEG 优先取真实样本）。
"""
import argparse
import io
import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mft_selftest import run, sha256_file, wsl_path  # noqa: E402

BUILDER = r"""#!/bin/bash
# 由 recovery_pipeline_selftest.py 生成：造一个装着真文件的 NTFS 卷，然后把它们全删掉
set -u
IMG="$1"; MNT="$2"; SRCDIR="$3"
rm -f "$IMG"
truncate -s 64M "$IMG" || { echo TRUNC_FAIL; exit 2; }
mkfs.ntfs -F -Q -L pipe "$IMG" >/dev/null 2>&1 || { echo MKFS_FAIL; exit 2; }
mkdir -p "$MNT"
LOOP=$(losetup -f --show "$IMG" 2>/dev/null) || { echo LOOP_FAIL; exit 2; }
mount -t ntfs-3g "$LOOP" "$MNT" 2>/dev/null || { echo MOUNT_FAIL; exit 2; }
cd "$SRCDIR" || exit 2
mkdir -p "$MNT/gone"
for f in *; do
  [ -f "$f" ] || continue
  cp "$f" "$MNT/gone/$f" || { echo COPY_FAIL; exit 2; }
done
sync
cd / || true
rm -rf "$MNT/gone"                     # 文件和目录一起删：模拟「误删」
sync
umount "$MNT" 2>/dev/null || umount -l "$MNT" 2>/dev/null || { echo UMFAIL; exit 2; }
losetup -d "$LOOP" 2>/dev/null
echo BUILD_OK
"""

RD16 = lambda b, o: b[o] | (b[o + 1] << 8)          # noqa: E731
RD32 = lambda b, o: struct.unpack_from("<I", b, o)[0]  # noqa: E731
RD64 = lambda b, o: struct.unpack_from("<Q", b, o)[0]  # noqa: E731


def make_sources(src_dir):
    """造 6 个真文件；JPEG 优先用 F: 上的真实样本（真编码器的 marker 更刁钻）"""
    os.makedirs(src_dir, exist_ok=True)
    from PIL import Image
    made = []
    real_jpgs = []
    cand = r"F:\weibo_cos\images"
    if os.path.isdir(cand):
        for fn in sorted(os.listdir(cand)):
            if fn.lower().endswith(".jpg"):
                real_jpgs.append(os.path.join(cand, fn))
    for i, (name, size) in enumerate((("photo1.jpg", (320, 240)), ("photo2.jpg", (200, 300)))):
        dst = os.path.join(src_dir, name)
        if i < len(real_jpgs):
            shutil.copyfile(real_jpgs[i], dst)
        else:
            Image.new("RGB", size, (30 * i, 90, 200 - 40 * i)).save(dst, "JPEG", quality=88)
        made.append(dst)
    for i, name in enumerate(("pic1.png", "pic2.png")):
        dst = os.path.join(src_dir, name)
        img = Image.new("RGBA", (160 + 40 * i, 120), (200, 40 * i, 90, 255))
        for x in range(0, img.width, 7):
            for y in range(0, img.height, 5):
                img.putpixel((x, y), (x % 256, y % 256, (x * y) % 256, 255))
        img.save(dst, "PNG")
        made.append(dst)
    # 注意：这里所有文件都要做成「非驻留」（>1KB）。小于约 700 字节的文件 NTFS 会把内容
    # 直接塞进 $MFT 记录的 $DATA 里，记录末尾 2 字节被 fixup 值替换 —— 那种文件只有过 fixup 的
    # mft_recover 能字节精确还原，纯签名扫描天生差那 2 字节。也就是说：
    #   MFT 在  -> 驻留/非驻留都能精确还原（mft_recover）
    #   MFT 毁  -> 只有非驻留文件的内容还能靠签名捞（sector_carve）
    gif = os.path.join(src_dir, "anim.gif")
    frames = []
    for k in range(8):
        fr = Image.new("P", (160, 120), k * 24)
        for x in range(0, 160, 3):
            for y in range(0, 120, 2):
                fr.putpixel((x, y), (x + y + k * 7) % 256)
        frames.append(fr)
    frames[0].save(gif, "GIF", save_all=True, append_images=frames[1:], duration=80, loop=0)
    made.append(gif)
    # OOXML：docx/xlsx/pptx 本质是 zip，carve 靠里面的目录前缀（word/ xl/ ppt/）分类，
    # 所以这里用 zipfile 手工造结构合法的三种文件，既不用额外依赖，也能验到「自动识别」这一步。
    ooxml = {
        "report.docx": ("word/document.xml",
                        '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
                        'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>pipeline selftest</w:t>'
                        '</w:r></w:p></w:body></w:document>'),
        "sheet.xlsx": ("xl/workbook.xml",
                       '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/'
                       'spreadsheetml/2006/main"><sheets><sheet name="S1" sheetId="1"/></sheets></workbook>'),
        "slides.pptx": ("ppt/presentation.xml",
                        '<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/'
                        'presentationml/2006/main"><p:sldIdLst/></p:presentation>'),
    }
    for name, (inner, xml) in ooxml.items():
        dst = os.path.join(src_dir, name)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml",
                       '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                       'package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/>'
                       '</Types>')
            z.writestr(inner, xml)
            # 用不可压缩的随机字节撑到 2.5KB：deflate 会把重复文本压到几百字节，
            # 那样文件又会变回「驻留」，MFT 一毁内容就没了（第一版就是这样栽的）
            z.writestr(zipfile.ZipInfo("docProps/pad.bin"), os.urandom(2500),
                       compress_type=zipfile.ZIP_STORED)
        made.append(dst)

    zp = os.path.join(src_dir, "bundle.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("hello.txt", "pipeline selftest\n" * 400)
        z.writestr("inner/data.bin", bytes(range(256)) * 40)
        z.writestr("inner/more.bin", b"deadbeef" * 2000)
    made.append(zp)
    return made


def build_volume(case_dir):
    os.makedirs(case_dir, exist_ok=True)
    src_dir = os.path.join(case_dir, "src")
    files = make_sources(src_dir)
    img = os.path.join(case_dir, "pipe.img")
    mnt = os.path.join(case_dir, "mnt")
    script = os.path.join(case_dir, "_build.sh")
    with open(script, "w", newline="\n", encoding="utf-8") as f:
        f.write(BUILDER)
    os.makedirs(mnt, exist_ok=True)
    r = run(["wsl", "-u", "root", "--", "bash", wsl_path(script),
             wsl_path(img), wsl_path(mnt), wsl_path(src_dir)])
    if r.returncode != 0 or "BUILD_OK" not in r.stdout:
        raise RuntimeError("造卷失败: rc=%s out=%r err=%r" % (r.returncode, r.stdout[-300:], r.stderr[-300:]))
    return img, {os.path.basename(p): p for p in files}


def wipe_mft_runs(img):
    """把 $MFT 自己的数据段整段清零（只清 MFT，不动文件数据扇区）"""
    with open(img, "r+b") as f:
        boot = f.read(1024)
        bps = RD16(boot, 0x0B)
        spc = boot[0x0D]
        cluster = bps * spc
        rec_size = boot[0x40]
        rec_size = rec_size * cluster if rec_size < 128 else (1 << (256 - rec_size))
        mft_lcn = RD64(boot, 0x30)
        f.seek(mft_lcn * cluster)
        rec0 = bytearray(f.read(rec_size))
        usa_off = RD16(rec0, 4)
        usa_cnt = RD16(rec0, 6)
        for i in range(1, usa_cnt):
            pos = i * bps - 2
            rec0[pos:pos + 2] = rec0[usa_off + i * 2: usa_off + i * 2 + 2]
        runs = []
        o = RD16(rec0, 0x14)
        while o + 8 <= rec_size:
            t = RD32(rec0, o)
            ln = RD32(rec0, o + 4)
            if t == 0xFFFFFFFF or ln == 0 or o + ln > rec_size:
                break
            if t == 0x80 and rec0[o + 8] != 0 and rec0[o + 9] == 0:
                ro = RD16(rec0, o + 0x20)
                lcn = 0
                p = o + ro          # 关键：runlist 偏移是相对属性起点，不是相对记录起点
                while p < o + ln:
                    hdr = rec0[p]
                    if hdr == 0:
                        break
                    ls, osz = hdr & 0xF, (hdr >> 4) & 0xF
                    if ls == 0:
                        break
                    rl = int.from_bytes(rec0[p + 1:p + 1 + ls], "little")
                    if osz:
                        roff = int.from_bytes(rec0[p + 1 + ls:p + 1 + ls + osz], "little", signed=True)
                        lcn += roff
                        runs.append((lcn, rl))
                    p += 1 + ls + osz
                break
            o += ln
        wiped = 0
        for lcn, rl in runs:
            f.seek(lcn * cluster)
            n = rl * cluster
            f.write(b"\x00" * n)
            wiped += n
    return wiped


def collect(dirpath):
    out = []
    for root, _d, files in os.walk(dirpath):
        for fn in files:
            p = os.path.join(root, fn)
            out.append((os.path.relpath(p, dirpath), p, sha256_file(p), os.path.getsize(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "recovery_pipeline_selftest"))
    ap.add_argument("--exe-dir", default=HERE)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    mft = os.path.join(args.exe_dir, "mft_recover.exe")
    carve = os.path.join(args.exe_dir, "sector_carve.exe")
    for exe in (mft, carve):
        if not os.path.isfile(exe):
            print("找不到 %s；先编译 tools 里的工具" % exe)
            return 2

    case = args.workdir
    shutil.rmtree(case, ignore_errors=True)
    print("=" * 74)
    print("步骤 1/4 造卷：64MB 真 NTFS + 9 个真文件（2 JPEG / 2 PNG / GIF / ZIP / docx / xlsx / pptx），然后整个删掉")
    print("=" * 74)
    try:
        img, originals = build_volume(case)
    except Exception as e:  # noqa: BLE001
        print("  SKIP 造卷不可用: %s" % e)
        return 0
    print("  镜像 %s，源文件 %d 个" % (img, len(originals)))
    want = {sha256_file(p): name for name, p in originals.items()}
    fail = 0

    print("\n" + "=" * 74)
    print("步骤 2/4 分支 A：MFT 还活着 -> mft_recover（应 9/9 字节精确、带目录树）")
    print("=" * 74)
    out_a = os.path.join(case, "out_mft")
    r = run([mft, img, out_a])
    print("  rc=%d" % r.returncode)
    got_a = collect(out_a)
    hit = sum(1 for _rel, _p, h, _s in got_a if h in want)
    print("  恢复 %d 个文件，其中 %d 个与原始 sha256 一致" % (len(got_a), hit))
    for rel, _p, h, sz in sorted(got_a):
        print("    %-40s %7d B  %s" % (rel, sz, want.get(h, "(内容不匹配!)")))
    if hit != len(want):
        print("  FAIL 分支 A 只对上 %d/%d 个文件" % (hit, len(want)))
        fail += 1

    print("\n" + "=" * 74)
    print("步骤 3/4 分支 B：把 $MFT 数据区整段清零（模拟 MFT 被毁）")
    print("=" * 74)
    wiped_img = os.path.join(case, "pipe_mftwiped.img")
    shutil.copyfile(img, wiped_img)
    n = wipe_mft_runs(wiped_img)
    print("  已清零 $MFT 数据区 %d 字节（文件数据扇区不动）" % n)
    out_b = os.path.join(case, "out_mft_after_wipe")
    r = run([mft, wiped_img, out_b])
    got_b = collect(out_b)
    print("  mft_recover rc=%d，导出 %d 个（元数据没了，这里就该是 0）" % (r.returncode, len(got_b)))
    if r.returncode != 0 or got_b:
        print("  FAIL 分支 B 的 mft_recover 行为不对（应 rc=0 且导出 0 个）")
        fail += 1

    print("\n" + "=" * 74)
    print("步骤 4/4 分支 B：sector_carve（靠签名把内容捞回来）")
    print("=" * 74)
    out_c = os.path.join(case, "out_carve")
    rc = run([carve, wiped_img, out_c])
    print("  rc=%d" % rc.returncode)
    got_c = collect(out_c)
    hit_c = {}
    for rel, _p, h, sz in got_c:
        if h in want:
            hit_c[want[h]] = (rel, sz)
    print("  扫出 %d 个文件，其中 %d 个与原始 sha256 一致" % (len(got_c), len(hit_c)))
    exts = {os.path.splitext(rel)[1].lower() for rel, _p, _h, _s in got_c}
    want_exts = {".docx", ".xlsx", ".pptx"}
    if want_exts.issubset(exts):
        print("  OK   OOXML 自动识别：docx/xlsx/pptx 都按 zip 内容正确改名")
    else:
        print("  FAIL OOXML 识别不全：期望 %s，实际 %s" % (sorted(want_exts), sorted(exts)))
        fail += 1
    for name in sorted(originals):
        if name in hit_c:
            print("    OK   %-16s -> %s" % (name, hit_c[name][0]))
        else:
            print("    MISS %-16s" % name)
    if len(hit_c) != len(want):
        print("  FAIL 分支 B 只捞回 %d/%d 个文件" % (len(hit_c), len(want)))
        print("       （注意：驻留文件的内容存在 $MFT 记录里，MFT 一旦被毁，签名扫描本就救不回来；")
        print("        夹具里的文件都做成了非驻留，所以这里应该 9/9）")
        fail += 1

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    if fail == 0:
        print("  PASS  文档里的决策链成立：MFT 在 -> mft_recover 精确还原；MFT 毁 -> sector_carve 捞内容")
    else:
        print("  FAIL  %d 项不符合预期" % fail)
    if not args.keep and fail == 0:
        shutil.rmtree(case, ignore_errors=True)
    elif args.keep:
        print("  现场保留在 %s" % case)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
