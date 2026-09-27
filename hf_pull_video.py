# -*- coding: utf-8 -*-
"""hf_pull_video.py — 用 hf-mirror 拉视频模型的小文件，并验证完整性

背景（实测）:
    · huggingface.co 本机**超时**（被墙）；hf-mirror.com **200 可用**。
      所以任何 HF 下载都必须设 HF_ENDPOINT=https://hf-mirror.com。
    · 本地 Wan2.1-T2V-1.3B 里那 3 个 15 B 文件（configuration_wan.py /
      modeling_wan.py / tokenizer_config.json）内容是字面量 "Entry not found"，
      是 **ModelScope 的占位符**；查 HF 仓库的文件清单，**它们本来就不存在**
      （HF 侧只有 config.json + google/umt5-xxl/tokenizer*）。
      ⇒ 不是"下载坏了"，是**两个平台的仓库结构不同**。

本脚本做的事:
    从 hf-mirror 拉 HF 仓库里**真正存在**的小文件，逐个校验：
    文件大小 > 0、HTTP 200、json 能解析（若是 json）。
    同时对已存在的大权重做一次**完整性抽检**（safetensors 头能否解析），
    确认本地那份不是残缺的。

用法:
    python hf_pull_video.py                 # 拉 Wan2.1 的配置/tokenizer
    python hf_pull_video.py --check-only    # 只校验本地，不下载
"""
from __future__ import annotations

import json
import os
import struct
import sys
import urllib.request

REPO = os.environ.get("HF_REPO", "Wan-AI/Wan2.1-T2V-1.3B")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
DEST = os.environ.get("HF_DEST", r"D:\work\textmodel\Wan2.1-T2V-1.3B")

# HF 仓库里真实存在的小文件（依据 /api/models 的文件清单）
SMALL_FILES = [
    "config.json",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/special_tokens_map.json",
]
# 本地已有的大权重（只校验，不下载）
BIG_LOCAL = [
    "diffusion_pytorch_model.safetensors",
    "Wan2.1_VAE.pth",
]


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "hf_pull/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def pull(rel):
    url = "%s/%s/resolve/main/%s" % (ENDPOINT, REPO, rel)
    try:
        st, data = http_get(url)
    except Exception as e:
        return False, "HTTP 失败: %s: %s" % (type(e).__name__, e)
    if st != 200 or not data:
        return False, "HTTP %s / %d B" % (st, len(data))
    out = os.path.join(DEST, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    note = "%d B" % len(data)
    if rel.endswith(".json"):
        try:
            json.loads(data.decode("utf-8"))
            note += "  json 可解析 ✓"
        except Exception as e:
            note += "  ⚠ json 解析失败: %s" % e
    return True, note


def check_safetensors(path):
    """只读头部，验证 safetensors 结构完整（不加载全部张量）。"""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 100 * 1024 * 1024:
                return False, "header 长度异常: %d" % n
            hdr = json.loads(f.read(n).decode("utf-8"))
        hdr.pop("__metadata__", None)
        total = os.path.getsize(path)
        # 数据段应能容纳所有张量
        need = 8 + n + max(v["data_offsets"][1] for v in hdr.values())
        ok = need <= total
        return ok, ("%d 张量, header %d B, 需要 %d B / 实际 %d B%s"
                    % (len(hdr), n, need, total, "" if ok else "  ⚠ 数据段超出文件大小"))
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)


def main():
    check_only = "--check-only" in sys.argv
    print("=" * 84)
    print("hf-mirror 拉取视频模型小文件 + 本地权重完整性校验")
    print("=" * 84)
    print("endpoint = %s" % ENDPOINT)
    print("repo     = %s" % REPO)
    print("dest     = %s" % DEST)

    print("\n[1] 本地大权重完整性（只读头部）")
    for rel in BIG_LOCAL:
        p = os.path.join(DEST, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            print("  %-42s 不存在" % rel)
            continue
        sz = os.path.getsize(p)
        if rel.endswith(".safetensors"):
            ok, note = check_safetensors(p)
        else:
            # .pth 是 zip：只看魔术字节
            with open(p, "rb") as f:
                magic = f.read(4)
            ok = magic == b"PK\x03\x04"
            note = "zip 魔术 %s" % ("✓" if ok else "✗ %r" % magic)
        print("  %-42s %14d B  %s" % (rel, sz, note))

    if check_only:
        print("\n(--check-only: 不下载)")
        return 0

    print("\n[2] 从镜像拉取小文件")
    ok_all = True
    for rel in SMALL_FILES:
        ok, note = pull(rel)
        print("  [%s] %-42s %s" % ("OK " if ok else "FAIL", rel, note))
        ok_all = ok_all and ok

    print("\n[3] 拉取后的本地清单")
    for root, _dirs, files in os.walk(DEST):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, DEST)
            sz = os.path.getsize(p)
            flag = ""
            if sz <= 16:
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        txt = f.read().strip()
                    flag = "  <-- 占位符! 内容=%r" % txt[:40]
                except Exception:
                    pass
            print("  %-46s %14d B%s" % (rel, sz, flag))

    print("\n%s" % ("ALL OK" if ok_all else "有失败项"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
