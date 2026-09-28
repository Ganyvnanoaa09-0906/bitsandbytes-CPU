"""latent_chunk_store 零拷贝改动 + bf16 读回修复的验证。

判定标准：
  L1. 各 dtype（fp32/fp16/bf16）5D 视频形状 put→get 逐字节一致。
  L2. bf16 必须能读回（改动前 np.dtype('bfloat16') 直接 TypeError）。
  L3. 非连续输入张量也能正确存读。
  L4. 负控制：篡改盘上 1 字节后读回必须不同（证明比对不是永远 PASS）。
  L5. 峰值内存：put 大 latent 时增量应远小于张量本身（零拷贝的证据）。
"""
import os
import sys
import json
import shutil
import tempfile

import numpy as np
import torch

# Resolved relative to this file; the scratch tree goes under the system temp
# directory. These were hard-coded to D:\work\..., which tied the test to one box.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from latent_chunk_store import LatentChunkStore

ROOT = os.path.join(tempfile.gettempdir(), "_verify_latent")
R = []


def check(name, ok, detail=""):
    R.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def raw(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT, exist_ok=True)

    # ---- L1/L2：三种 dtype 的 5D 视频 latent ----
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        tag = str(dtype).replace("torch.", "")
        d = os.path.join(ROOT, tag)
        st = LatentChunkStore(d, keep=False)
        # 与 Wan / AnimateDiff 同形状的 5D latent (B, C, F, H, W)
        lat = torch.randn(1, 4, 16, 32, 32).to(dtype)
        ref = raw(lat)

        info = st.put("vid", lat)
        check(f"L1a put 返回字节数正确 [{tag}]",
              info["bytes"] > len(ref) and info["shape"] == [1, 4, 16, 32, 32],
              f"bytes={info['bytes']} 载荷={len(ref)}")

        try:
            back = st.get("vid")
            got = raw(back)
            check(f"L2 get 可读回 [{tag}]", True, f"shape={tuple(back.shape)} dtype={back.dtype}")
            check(f"L1b 逐字节一致 [{tag}]", got == ref and back.shape == (1, 4, 16, 32, 32),
                  f"{len(got)}B vs {len(ref)}B  shape={tuple(back.shape)}")
        except Exception as e:
            check(f"L2 get 可读回 [{tag}]", False, f"{type(e).__name__}: {e}")
        st.cleanup()

    # ---- L3：非连续输入 ----
    st = LatentChunkStore(os.path.join(ROOT, "noncontig"), keep=False)
    base = torch.randn(32, 16)
    nc = base.t()                       # 非连续
    assert not nc.is_contiguous()
    ref = raw(nc)
    st.put("nc", nc)
    back = st.get("nc")
    check("L3 非连续输入逐字节一致", raw(back) == ref, f"shape={tuple(back.shape)}")
    st.cleanup()

    # ---- L4：负控制 ----
    st = LatentChunkStore(os.path.join(ROOT, "neg"), keep=False)
    lat = torch.randn(1, 4, 8, 8, 8)
    ref = raw(lat)
    st.put("n", lat)
    p = st._paths["n"]
    with open(p, "rb") as f:
        blob = bytearray(f.read())
    # 改最后一个字节（一定落在数据区，不是 header）
    blob[-1] ^= 0xFF
    with open(p, "wb") as f:
        f.write(bytes(blob))
    # 清掉内存里的路径缓存与 mmap，强制重新读盘
    st._paths.clear()
    st.cleanup_maps = None
    st2 = LatentChunkStore(os.path.join(ROOT, "neg"), keep=True)
    back = st2.get("n")
    check("L4 篡改 1 字节后读回不同（负控制）", back is not None and raw(back) != ref,
          "比对方法确实能发现差异")

    # ---- L5：峰值内存 ----
    import psutil

    def rss():
        return psutil.Process().memory_info().rss / 1024 / 1024

    st = LatentChunkStore(os.path.join(ROOT, "mem"), keep=False)
    big = torch.randn(1, 16, 64, 128, 128)      # 64MB fp32
    mb = big.numel() * 4 / 1024 / 1024
    base = rss()
    peak = base
    for i in range(4):
        st.put(f"big{i}", big)
        peak = max(peak, rss())
    check("L5 零拷贝：峰值增量远小于张量本身", (peak - base) < mb * 0.5,
          f"张量 {mb:.0f}MB，峰值增量 {peak - base:.1f}MB")
    st.cleanup()

    npass = sum(1 for r in R if r["pass"])
    print(f"\n{'='*60}\n总计 {npass}/{len(R)} PASS\n{'='*60}")
    for r in R:
        if not r["pass"]:
            print(f"  FAIL: {r['check']}  {r['detail']}")
    out_dir = os.path.join(HERE, "cloud_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "verify_latent.json"), "w", encoding="utf-8") as f:
        json.dump({"results": R, "passed": npass, "total": len(R)}, f,
                  indent=2, ensure_ascii=False)
    shutil.rmtree(ROOT, ignore_errors=True)
    return 0 if npass == len(R) else 1


if __name__ == "__main__":
    sys.exit(main())
