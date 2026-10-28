# -*- coding: utf-8 -*-
"""test_video_e2e.py — 视频端到端管线（video_e2e）回归自检。

**不加载任何模型、不需要 torch**（只用 numpy + PIL + 标准库）。
覆盖四类：
  [A] 3 个历史 bug 的修复断言（E2E-PATCH 标记 + 行为断言）
  [B] temporal_metrics 在 synthetic 帧序列上的正确性（位移检测 / 边界拆分 / 统一 JSON schema）
  [C] video_e2e.py 的 dry-run：命令拼装正确（不执行模型）
  [D] 编排器不 import torch（内存红线：编排器不能自己持有模型）

跑法:  python test_video_e2e.py
退出码 0 = 全部 PASS。
（脚本风格对齐仓库既有 test_*.py：独立脚本 + 退出码，不依赖 pytest 配置。）
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import ast
import contextlib
import glob
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))     # 仓库根
PY = sys.executable
TMP = os.path.join(tempfile.gettempdir(), "video_e2e_selftest")
TOOLS_DUMMY = os.path.join(TMP, "tools")

RESULTS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag:4s}] {name}" + (f"  -- {detail}" if detail else ""))
    RESULTS.append((name, bool(cond), detail))
    return bool(cond)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def env_with_tools():
    env = dict(os.environ)
    env["VIDEO_E2E_TOOLS"] = TOOLS_DUMMY
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---------------------------------------------------------------------------
# synthetic 帧序列（PNG，无需 torch）
# ---------------------------------------------------------------------------
def _save(frames, d):
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f.astype(np.uint8)).save(os.path.join(d, f"f{i:03d}.png"))
    return d


def make_synthetic():
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(TOOLS_DUMMY, exist_ok=True)
    # 供 dry-run 用的占位训练工具（_tools_path 只检查存在性，不执行）
    open(os.path.join(TOOLS_DUMMY, "run_train_stages.py"), "w", encoding="utf-8").write(
        "# placeholder for video_e2e dry-run tests\n")

    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = ((xx * 1.3 + yy * 0.7) % 256)
    base_rgb = np.stack([base, (base * 0.8) % 256, (base * 1.1) % 256], axis=2)

    # 1) 整体平移 +2px/帧（应判 has_global_motion=True）
    shift_dir = _save([np.roll(base_rgb, shift=2 * i, axis=1) for i in range(8)],
                      os.path.join(TMP, "shift"))

    # 2) 每帧独立噪声（mad 大但**不是**运动 -> has_global_motion=False）
    rng = np.random.default_rng(0)
    noise_dir = _save([rng.integers(0, 256, (H, W, 3)) for _ in range(8)],
                      os.path.join(TMP, "noise"))

    # 3) 2 chunks × 4 帧：chunk 内小噪声，seam（frame3->4）换内容
    a = rng.integers(60, 120, (H, W, 3))
    b = rng.integers(180, 240, (H, W, 3))
    boundary = [np.clip(a + rng.integers(-2, 3, (H, W, 3)), 0, 255) for _ in range(4)]
    boundary += [np.clip(b + rng.integers(-2, 3, (H, W, 3)), 0, 255) for _ in range(4)]
    boundary_dir = _save(boundary, os.path.join(TMP, "boundary"))
    return dict(shift=shift_dir, noise=noise_dir, boundary=boundary_dir,
                frame_paths=sorted(glob.glob(os.path.join(boundary_dir, "*.png"))))


# ---------------------------------------------------------------------------
# [A] 三个历史 bug 的修复断言
# ---------------------------------------------------------------------------
def test_bug1_blend_overlap():
    print("\n=== [A1] bug1: sliding_window.blend_overlap 空实现（已修）===")
    src = open(os.path.join(HERE, "sliding_window.py"), encoding="utf-8").read()
    check("E2E-PATCH[P3] 标记存在于 blend_overlap 说明中", "E2E-PATCH[P3]" in src)
    tree = ast.parse(src)
    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "blend_overlap":
            func = node
    body_nodes = [n for n in (func.body if func else [])
                  if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                          and isinstance(n.value.value, str))]
    has_arith = any(isinstance(n, (ast.Assign, ast.For, ast.BinOp)) for n in body_nodes)
    check("函数体不再是 `return frames` 空实现（含循环/算术）",
          func is not None and has_arith,
          f"body={[type(n).__name__ for n in body_nodes]}")

    sw = load_module(os.path.join(HERE, "sliding_window.py"), "sw_port")
    prev = [np.zeros((4, 4, 3), dtype=np.uint8)]
    head = [np.full((4, 4, 3), 100, dtype=np.uint8)]
    out = sw.blend_overlap(prev, head)
    check("blend_overlap 真的混合（0 与 100 -> 50）",
          len(out) == 1 and int(out[0].mean()) == 50, f"out_mean={int(out[0].mean())}")
    check("混合输出既不是 prev 也不是 head（不是空实现）",
          int(out[0].mean()) != 0 and int(out[0].mean()) != 100)


def test_bug2_lora_prefix():
    print("\n=== [A2] bug2: LoRA 手动 merge 只认 base_model.model. 前缀（已修）===")
    src = open(os.path.join(HERE, "independent_chunks.py"), encoding="utf-8").read()
    check("E2E-PATCH[P1] 标记存在（双前缀归一化 + 0 命中报错）", "E2E-PATCH[P1]" in src)
    # P6 回归：生成脚本必须把 context 按帧展开（否则 diffusers 0.40 直接崩，见 Smoke A 定位）
    sw_src = open(os.path.join(HERE, "sliding_window.py"), encoding="utf-8").read()
    p6_ok = ("E2E-PATCH[P6]" in src and "E2E-PATCH[P6]" in sw_src
             and "repeat_interleave" in src and "repeat_interleave" in sw_src)
    check("E2E-PATCH[P6] context 按帧展开（两个生成脚本都含 repeat_interleave）", p6_ok)

    ic = load_module(os.path.join(HERE, "independent_chunks.py"), "ic_port")
    sd = {
        "base_model.model.down_blocks.0.attn1.to_q.lora_A.default.weight": np.zeros((1,)),
        "base_model.model.down_blocks.0.attn1.to_q.lora_B.default.weight": np.zeros((1,)),
        "unet.mid_block.attn2.to_v.lora_A.weight": np.zeros((1,)),
        "unet.mid_block.attn2.to_v.lora_B.weight": np.zeros((1,)),
        "some_other.weight": np.zeros((1,)),
    }
    pairs = ic.iter_lora_pairs(sd)
    check("两种前缀都被归一化并配对（2 对）", len(pairs) == 2, f"pairs={sorted(pairs)}")
    old = {}
    for k, v in sd.items():
        if k.startswith("base_model.model."):
            rest = k[len("base_model.model."):]
            if rest.endswith(".lora_A.default.weight"):
                old.setdefault(rest[:-len(".lora_A.default.weight")], {})["A"] = v
            elif rest.endswith(".lora_B.default.weight"):
                old.setdefault(rest[:-len(".lora_B.default.weight")], {})["B"] = v
    old = {p: ab for p, ab in old.items() if "A" in ab and "B" in ab}
    check("旧逻辑同样输入只命中 1 对（unet.* 被静默丢弃）",
          len(old) == 1 and len(pairs) > len(old), f"old={len(old)} new={len(pairs)}")
    check("0 命中会 SystemExit（applied == 0 断言存在）", "applied == 0" in src)


def test_bug3_boundary():
    print("\n=== [A3] bug3: overlap=0 时边界拆分不执行（已修）===")
    src = open(os.path.join(HERE, "temporal_metrics.py"), encoding="utf-8").read()
    check("E2E-PATCH[P3]/[P3b] 标记存在（条件 + 索引修复）",
          "E2E-PATCH[P3]" in src and "E2E-PATCH[P3b]" in src)
    t = make_synthetic()
    paths = t["frame_paths"]
    tm = load_module(os.path.join(HERE, "temporal_metrics.py"), "tm_port")

    # 旧 API 必须保留（scan_cn_one.py / sweep_carry_noise.py 会 import）
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        legacy = tm.analyze(paths, "boundary", chunk=4, overlap=0)
    check("旧接口 analyze() 保留（可 import、返回 mad 等原字段）",
          isinstance(legacy, dict) and "mad" in legacy and "warp" in legacy)
    res = tm.analyze_sequence(paths, "boundary", chunk=4, overlap=0)
    bd = res["boundary"]
    check("overlap=0 也执行边界拆分（boundary 非 None）", bd is not None)
    check("boundary 索引在 chunk-1=3（seam 位置正确）",
          bd is not None and 3 in bd["indices"], f"indices={bd['indices'] if bd else None}")
    check("合成硬切 boundary/intra 比值 > 2（能抓到 seam）",
          bd is not None and bd["ratio"] is not None and bd["ratio"] > 2,
          f"ratio={bd['ratio'] if bd else None}")


# ---------------------------------------------------------------------------
# [B] synthetic 正确性 + JSON schema
# ---------------------------------------------------------------------------
def test_metrics_correctness():
    print("\n=== [B] temporal metrics：位移检测 / 噪声不误判 / JSON schema ===")
    t = make_synthetic()
    tm = load_module(os.path.join(HERE, "temporal_metrics.py"), "tm_port2")
    r_shift = tm.analyze_sequence(sorted(glob.glob(os.path.join(t["shift"], "*.png"))), "shift")
    te = r_shift["temporal"]
    check("整体平移序列：has_global_motion=True", te["has_global_motion"] is True,
          f"warp/mad={te['warp_over_mad']:.3f} shift={te['best_shift']}")
    check("整体平移序列：best_shift 位移量≈2（np.roll 符号约定，取绝对值）",
          abs(abs(te["best_shift"]["dx"]) - 2) <= 1, f"best_shift={te['best_shift']}")
    check("整体平移序列：warp_err 明显小于 mad", te["warp_err"] < 0.85 * te["mad"],
          f"warp={te['warp_err']:.3f} mad={te['mad']:.3f}")
    r_noise = tm.analyze_sequence(sorted(glob.glob(os.path.join(t["noise"], "*.png"))), "noise")
    te2 = r_noise["temporal"]
    check("纯噪声序列：mad 很大但 has_global_motion=False（不把噪声当运动）",
          te2["mad"] > 10 and te2["has_global_motion"] is False,
          f"mad={te2['mad']:.1f} motion={te2['has_global_motion']}")

    js = os.path.join(TMP, "metrics_boundary.json")
    p = subprocess.run([PY, os.path.join(HERE, "temporal_metrics.py"), t["boundary"],
                        "--chunk", "4", "--overlap", "0", "--json-out", js,
                        "--run-id", "selftest"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    check("CLI 正常退出", p.returncode == 0, f"rc={p.returncode} err={p.stderr[-200:]}")
    m = json.load(open(js, encoding="utf-8")) if os.path.exists(js) else {}
    check("metrics.json schema == video_e2e/1", m.get("schema") == "video_e2e/1",
          f"schema={m.get('schema')}")
    check("metrics.json 有 frames/temporal/white/detail/boundary/verdict 段",
          all(k in m for k in ("frames", "temporal", "white", "detail", "boundary", "verdict")),
          f"keys={sorted(m)}")
    check("boundary 段带 mode/ratio/note、white 段带 r2",
          "ratio" in (m.get("boundary") or {}) and "note" in (m.get("boundary") or {})
          and "r2" in (m.get("white") or {}))
    check("frames.count == 8 且 all_finite",
          m["frames"]["count"] == 8 and m["frames"]["all_finite"] is True)


# ---------------------------------------------------------------------------
# [C] dry-run：命令拼装（不执行模型）
# ---------------------------------------------------------------------------
def test_dry_run():
    print("\n=== [C] video_e2e.py dry-run：命令拼装（不执行模型） ===")
    orch = os.path.join(HERE, "video_e2e.py")
    run_gen = os.path.join(TMP, "run_dry_gen")
    cases = [
        (["gen", "--mode", "independent", "--res", "128", "--frames", "16", "--chunk", "8",
          "--steps", "6", "--out", run_gen, "--dry-run"], "independent_chunks.py"),
        (["gen", "--mode", "sliding", "--res", "128", "--frames", "26", "--chunk", "8",
          "--overlap", "2", "--steps", "6", "--out", run_gen + "_slide", "--dry-run"],
         "sliding_window.py"),
        (["smoke", "--profile", "infer", "--root", os.path.join(TMP, "smoke_infer"),
          "--dry-run"], "gen"),
        (["smoke", "--profile", "full", "--root", os.path.join(TMP, "smoke_full"),
          "--dry-run"], "independent_chunks.py"),
    ]
    for argv, want in cases:
        p = subprocess.run([PY, orch] + argv, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env_with_tools())
        ok = p.returncode == 0 and want in p.stdout
        check(f"dry-run {' '.join(argv[:2])} -> 含 {want}", ok,
              f"rc={p.returncode}" + ("" if ok else f" out={p.stdout[-300:]}"))
    cfg = os.path.join(run_gen, "config.json")
    plan = os.path.join(run_gen, "plan.json")
    check("gen --dry-run 写出 config.json/plan.json（可复现凭据）",
          os.path.exists(cfg) and os.path.exists(plan))
    if os.path.exists(plan):
        pj = json.load(open(plan, encoding="utf-8"))
        check("plan 记录 chunks/produced_frames",
              pj.get("chunks") == 2 and pj.get("produced_frames") == 16, f"plan={pj}")
    else:
        check("plan 记录 chunks/produced_frames", False, "plan.json 缺失")
    p = subprocess.run([PY, orch, "gen", "--mode", "sliding", "--res", "128", "--chunk", "8",
                        "--overlap", "2", "--lora", "x.safetensors",
                        "--out", os.path.join(TMP, "run_bad"), "--dry-run"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env_with_tools())
    check("sliding+LoRA 被显式拒绝（不会静默忽略 LoRA）",
          p.returncode != 0 and "没有 LoRA 加载入口" in (p.stdout + p.stderr))

    more = [
        (["metrics", "--out", run_gen, "--chunk", "8", "--dry-run"], "temporal_metrics.py"),
        (["post", "--out", run_gen, "--dry-run"], "post_process.py"),
        (["train", "--data", os.path.join(TMP, "dummy.pt"),
          "--out", os.path.join(TMP, "run_dry_train"), "--dry-run"], "run_train_stages.py"),
    ]
    for argv, want in more:
        p = subprocess.run([PY, orch] + argv, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env_with_tools())
        ok = p.returncode == 0 and want in p.stdout
        check(f"dry-run {argv[0]} -> 含 {want}", ok,
              f"rc={p.returncode}" + ("" if ok else f" out={p.stdout[-300:]}"))

    sys.path.insert(0, HERE)
    from scripts.lib import video_e2e
    try:
        v_e2e.normalize_lora_path(os.path.join(TMP, "definitely_missing.safetensors"))
        raised = False
    except SystemExit as e:
        raised = "definitely_missing.safetensors" in str(e)
    check("normalize_lora_path：不存在的 LoRA 文件 → SystemExit 且报出路径", raised)

    fake_pt = os.path.join(TMP, "fake_ckpt.pt")
    with open(fake_pt, "wb") as fh:
        fh.write(b"x" * 8)
    p = subprocess.run([PY, orch, "smoke", "--profile", "full", "--root",
                        os.path.join(TMP, "smoke_full_fromckpt"),
                        "--from-ckpt", fake_pt, "--dry-run"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env_with_tools())
    out = p.stdout
    check("smoke full --from-ckpt：跳过训练段（命令里无 run_train_stages.py）",
          p.returncode == 0 and "run_train_stages.py" not in out, f"rc={p.returncode}")
    check("smoke full --from-ckpt：命令里出现 convert/绝对路径 lora",
          os.path.abspath(fake_pt) in out or "convert_lora_ckpt.py" in out)


# ---------------------------------------------------------------------------
# [D] 编排器不 import torch
# ---------------------------------------------------------------------------
def test_no_torch():
    print("\n=== [D] 编排器不 import torch（内存红线：编排器不能自己持有模型） ===")
    src = open(os.path.join(HERE, "video_e2e.py"), encoding="utf-8").read()
    check("静态：video_e2e.py 无 'import torch'",
          re.search(r"^\s*import torch", src, re.M) is None)
    code = ("import sys; sys.path.insert(0, r'%s'); import video_e2e; "
            "print('TORCH_LOADED' if 'torch' in sys.modules else 'NO_TORCH')" % HERE)
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    check("动态：import video_e2e 后 sys.modules 里没有 torch",
          "NO_TORCH" in p.stdout, f"out={p.stdout.strip()} err={p.stderr[-200:]}")


def summary():
    print("\n" + "=" * 74)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"video_e2e selftest: {len(RESULTS)-n_fail}/{len(RESULTS)} PASS, {n_fail} FAIL")
    if n_fail:
        print("FAILED:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}  ({detail})")
    print("=" * 74)
    return 0 if n_fail == 0 else 1


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if os.path.isdir(TMP):
        import shutil
        shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    test_bug1_blend_overlap()
    test_bug2_lora_prefix()
    test_bug3_boundary()
    test_metrics_correctness()
    test_dry_run()
    test_no_torch()
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
