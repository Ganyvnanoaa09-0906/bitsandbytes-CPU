"""align_sec9.py -- bring QUICKSTART.md section 9 in line with its English counterpart.

The Chinese section 9 had been edited to a different vocabulary from the English
one, leaving the two out of step:

    zh: 9. 数据恢复            / 9.2 自带的 sector_mirror        / 9.3 Windows File Recovery
    en: 9. Disaster Recovery   / 9.2 Tier 1: this repo's ...     / 9.3 Tier 2: ...

and one of the edits removed the clause that 9.3's own body still refers to, while
9.4 kept citing "第一梯队" after 9.2's heading stopped defining it.

The instruction is that the CHINESE side moves to match the English one, so each
line below is set to an exact rendering of its English counterpart, keeping the
file's existing mixed zh/en heading style ("## 9. 灾难恢复（Data Recovery）").

Every replacement is content-addressed and must match exactly once; a stale line
fails loudly rather than editing the wrong text.

Usage:
    python align_sec9.py <QUICKSTART.md> [--apply]
"""
import io
import os
import sys

FIXES = [
    # heading: English is "## 9. Disaster Recovery (Data Recovery)"
    ("## 9. 数据恢复（Data Recovery）",
     "## 9. 灾难恢复（Disaster Recovery）"),
    # English: "### 9.2 Tier 1: this repo's `sector_mirror` (bypass the filesystem)"
    ("### 9.2 自带的 `sector_mirror`",
     "### 9.2 第一梯队：本仓库自带的 `sector_mirror`（绕过文件系统）"),
    # English: "### 9.3 Tier 2: Windows File Recovery (when repo/tool unavailable)"
    ("### 9.3 Windows File Recovery",
     "### 9.3 第二梯队：本仓库 / 工具均不可用时 → Windows File Recovery"),
    # the edit removed "连本仓库 /" and left a stray space; English says "repo/tool unavailable"
    ("> **适用**：**MFT 损坏过于严重， Python / exe 都无法运行**。退回微软商店的",
     "> **适用**：**MFT 损坏过于严重，连本仓库 / Python / exe 都无法运行**。退回微软商店的"),
    # English: "**Mirror first (Tier 1) then operate** on the healthy copy;"
    ("2. **优先镜像（第一梯队）再操作**——在完好副本上恢复最安全；",
     "2. **优先镜像（第一梯队 Tier 1）再操作**——在完好副本上恢复最安全；"),
]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    apply_changes = "--apply" in sys.argv
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()

    ok = True
    for old, new in FIXES:
        n = text.count(old)
        if n != 1:
            print(f"  SKIP ({n} match(es), expected 1): {old[:70]!r}")
            ok = False
            continue
        text = text.replace(old, new)
        print(f"  edited")
        print(f"      was: {old}")
        print(f"      now: {new}")

    if not ok:
        print("\nFAIL: at least one pattern did not match exactly once; nothing written")
        return 1

    if apply_changes:
        tmp = path + ".tmp"
        with io.open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
        print(f"\nAPPLIED -> {path}")
    else:
        print("\ndry run only; pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
