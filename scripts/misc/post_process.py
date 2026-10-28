# -*- coding: utf-8 -*-
"""post_process.py — 最终后处理（E2E 管线版，2026-10-23 port）。

本文件由 E2E 管线副本 port 回仓库（feat/video-e2e，起点 824995d0）。
冻结决策未变：FSRCNN ×4 细节恢复 + area 抽回原尺寸。改动见下方 E2E-PATCH 列表。
冻结决策：**FSRCNN ×4 细节恢复 + `area` 抽回原尺寸**（Real-ESRGAN 会"塑料化"且慢 ~70×）。

E2E-PATCH 列表（详见 PENDING_REPO_PORT.md）：
  [P1] 加 `--verify`：mp4 读回校验（帧数 + 尺寸；从 anime_video_e2e.py L217–230 移植）。
       只写文件不校验 = 不算成功。
  [P2] 加 `--restorer {fsrcnn,esrgan6b,none}`，默认 `fsrcnn`（冻结决策）；
       `esrgan6b` 走 sr_cost.load（6 块模型；23 块在 256 输入下会段错误，VIDEO_GUIDE §5.2）。
  [P3] 输出布局对齐管线 run 目录：`<out>/frames_sr/fNNN.png`、`<out>/video.gif`、`<out>/video.mp4`；
       `<name>.json` 保留（名字可由 --name 覆盖）。
  [P4] 惰性 import（torch / av / detail_restore / sr_cost 只在真正处理时导入）。
  [P5] 加 `--fps` 显式、`--limit-s` 单进程保险丝（逐帧边界检查）。

用法:
  python post_process_e2e.py --src runs/x/frames --out runs/x --verify --fps 8
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def _reconfigure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def lapvar(a):
    a = np.asarray(a).astype(np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def write_mp4(frames, path, fps):
    """Encode with PyAV. Returns True on success. (PyAV 本机已装: av 16.0.0)"""
    try:
        import av
    except ImportError:
        print(f"[warn] `av` not available -> skipping {os.path.basename(path)}")
        return False
    from fractions import Fraction
    try:
        out = av.open(path, "w")
        st = out.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
    except Exception:
        out = av.open(path, "w")
        st = out.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1000))
    st.width, st.height = frames[0].shape[1], frames[0].shape[0]
    st.pix_fmt = "yuv420p"
    st.bit_rate = 16_000_000
    for fr in frames:
        for p in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    print(f"[save] {path}  ({len(frames)} frames @ {fps}fps, {len(frames)/fps:.2f}s)")
    return True


def verify_mp4(path, expect_frames):
    """E2E-PATCH[P1]: 读回校验（帧数 + 尺寸），返回 dict。"""
    try:
        import av
        c = av.open(path)
        st = c.streams.video[0]
        n = sum(1 for _ in c.decode(video=0))
        w, h = st.width, st.height
        c.close()
        ok = (n == expect_frames)
        print(f"[verify] 读回 {n} 帧（期望 {expect_frames}），尺寸 {w}x{h}  "
              f"{'OK' if ok else 'MISMATCH'}")
        return dict(path=path, frames=int(n), size=[int(w), int(h)], ok=bool(ok))
    except Exception as e:
        print(f"[verify] 读回失败: {type(e).__name__}: {e}")
        return dict(path=path, frames=0, size=None, ok=False, error=str(e))


def main():
    _reconfigure_stdout()
    ap = argparse.ArgumentParser(description="post-process (E2E copy): restorer -> gif/mp4")
    ap.add_argument("--src", required=True, help="frames dir, or its parent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="video")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--restorer", default="fsrcnn", choices=["fsrcnn", "esrgan6b", "none"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit-s", type=float, default=0.0)
    a = ap.parse_args()

    src = a.src
    if not glob.glob(os.path.join(src, "f*.png")):
        cand = os.path.join(src, "frames")
        if glob.glob(os.path.join(cand, "f*.png")):
            src = cand
    ps = sorted(glob.glob(os.path.join(src, "f*.png")),
                key=lambda p: int(re.search(r"f(\d+)\.png$", p).group(1)))
    if not ps:
        print(f"[err] no fNNN.png under {src}")
        return 2

    t_all = time.time()
    fdir = os.path.join(a.out, "frames_sr")
    os.makedirs(fdir, exist_ok=True)
    print(f"[cfg] {len(ps)} frames  src={src}")
    print(f"[cfg] out={a.out}  name={a.name}  fps={a.fps}  restorer={a.restorer}")

    from PIL import Image
    laps_before, laps_after = [], []

    if a.restorer == "fsrcnn":
        # E2E-PATCH[P4]: 惰性 import + 惰性加载权重
        import torch
        import torch.nn.functional as F
        from scripts.lib.detail_restore import load_sr
        torch.set_num_threads(a.threads)
        sr = load_sr()
        print("[cfg] FSRCNN x4 -> area decimate")
    else:
        import torch
        import torch.nn.functional as F
        from scripts.lib.sr_cost import ANIME_6B, load as load_rrdb
        torch.set_num_threads(a.threads)
        if a.restorer == "esrgan6b":
            # 6 块模型：23 块在 256 输入下段错误（VIDEO_GUIDE §5.2）
            model, npar, miss, unexp = load_rrdb(ANIME_6B, 6)
            print(f"[cfg] RealESRGAN anime_6B  {npar/1e6:.2f}M 参数 missing={miss} unexpected={unexp}")
            sr = model
        else:
            sr = None          # restorer=none：只编码

    imgs = []
    for i, p in enumerate(ps):
        if a.limit_s and (time.time() - t_all) > a.limit_s:
            print(f"[limit] 超时 {a.limit_s}s，在帧 {i} 退出")
            break
        pil = Image.open(p).convert("RGB")
        laps_before.append(lapvar(pil))
        if sr is None:
            arr = np.asarray(pil).astype(np.uint8)
        else:
            t = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1
                                                                 ).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                up = sr(t)
                if a.restorer == "fsrcnn":
                    dn = F.interpolate(up, size=pil.size[::-1], mode="area").clamp(0, 1)
                else:
                    # Real-ESRGAN 输出已是 ×4 目标尺寸
                    dn = up.clamp(0, 1)
                    dn = F.interpolate(dn, size=pil.size[::-1], mode="area").clamp(0, 1) \
                        if list(dn.shape[-2:]) != [pil.size[1], pil.size[0]] else dn
            arr = (dn[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        laps_after.append(lapvar(arr))
        im = Image.fromarray(arr)
        im.save(os.path.join(fdir, f"f{i:03d}.png"))
        imgs.append(im)
        if (i + 1) % 10 == 0 or i == len(ps) - 1:
            print(f"  {i+1}/{len(ps)}  lap {laps_before[-1]:.1f} -> {laps_after[-1]:.1f}",
                  flush=True)

    dt = time.time() - t_all
    gif = os.path.join(a.out, f"{a.name}.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000 / a.fps)), loop=0)
    print(f"[save] {gif}  ({len(imgs)} frames @ {a.fps}fps)")
    mp4 = os.path.join(a.out, f"{a.name}.mp4")
    mp4_ok = write_mp4([np.asarray(im) for im in imgs], mp4, a.fps)
    ver = None
    if a.verify and mp4_ok:
        ver = verify_mp4(mp4, len(imgs))

    mb = float(np.mean(laps_before))
    ma = float(np.mean(laps_after))
    print(f"\n{'='*64}")
    print(f"RESULT  {len(ps)} frames in {dt:.1f}s ({dt/len(ps):.2f}s/frame)")
    print(f"  Laplacian mean : {mb:.1f} -> {ma:.1f}  (x{(ma/mb) if mb else float('nan'):.2f})")
    print(f"  frames dir     : {fdir}")
    res = dict(src=src, restorer=a.restorer, decimate="area", fps=a.fps,
               n=len(ps), lap_before=laps_before, lap_after=laps_after,
               lap_mean_before=mb, lap_mean_after=ma, seconds=dt,
               mp4=mp4_ok, verify=ver, frames_dir=fdir,
               out_size=[int(imgs[0].width), int(imgs[0].height)])
    with open(os.path.join(a.out, f"{a.name}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"[save] {os.path.join(a.out, a.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
