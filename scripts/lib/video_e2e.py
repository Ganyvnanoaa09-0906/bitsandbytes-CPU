# -*- coding: utf-8 -*-
"""video_e2e.py — 视频端到端管线编排器（E2E 管线，2026-10-23 port 进仓库）。

设计文档：collab/shared/VIDEO_E2E_PIPELINE_PLAN.md（不在本仓库内）
组件脚本（本仓库根，已由 E2E 副本 port）：sliding_window.py / independent_chunks.py /
  temporal_metrics.py / post_process.py
训练工具仍是外部脚本（video_mvp/），通过环境变量 VIDEO_E2E_TOOLS 或默认 ../video_mvp 定位。

三条硬规则（全部来自既有实测/事故）：
  1. **绝不 import torch**：编排器只拼命令、跑 subprocess；任何模型都在子进程里加载
     （两个大模型进程并发会 0xC0000005，VIDEO_GUIDE §0 ①）。
  2. **每个 stage 一个进程**：生成/采样（~6.8–11 GB）→ 指标（<1 GB）→ 后处理（<3 GB），
     前一个退出后才起下一个。
  3. **PASS/FAIL 看具体数值**，不看 exit 0：metrics.json 的字段 + mp4 读回 + 帧数断言。

子命令：
  gen      生成（independent | sliding）→ frames/fNNN.png →（默认）指标 + 后处理
  train    LoRA 训练（分段 resume）+ 可选转换 diffusers LoRA
  metrics  只重算指标（合并 gen_meta/chunk_metrics → 统一 metrics.json）
  post     只做后处理（FSRCNN ×4 → gif/mp4 + 读回校验）
  smoke    验收 smoke：infer / train / full（默认执行；--dry-run 只打印命令）
  report   把 run 目录汇总成 markdown（PASS/FAIL + 指标表）

Phase 1 自检只跑：`--help`、`metrics` 的 synthetic 单测、各命令 `--dry-run`。
真正的 gen/train 由 lead 发窗口后执行（见 design doc §6）。

用法:
  python video_e2e.py gen --mode independent --res 128 --frames 16 --chunk 8 --steps 6 \
      --out D:\\work\\video_mvp\\runs\\smokeA
  python video_e2e.py smoke --profile infer --dry-run
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

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = HERE                      # 组件脚本已随 port 落到仓库根
REPO = HERE                     # stage 子进程 cwd = 仓库根
RUNS = os.path.join(HERE, "runs")
# 训练/数据缓存仍是外部工具（不随本仓库 port；见 PENDING_REPO_PORT.md §8）
TOOLS = os.environ.get("VIDEO_E2E_TOOLS", os.path.join(os.path.dirname(HERE), "video_mvp"))

def _tools_path(name):
    p = os.path.join(TOOLS, name)
    if not os.path.exists(p):
        raise SystemExit(f"[err] 找不到外部训练工具 {p}\n"
                         f"      设置 VIDEO_E2E_TOOLS 指向 video_mvp 目录（含 {name}）")
    return p
PY = sys.executable
SCHEMA = "video_e2e/1"

THREADS_DEFAULT = 4   # 本机 4500U：4 线程优于 6 线程（既有实测：Qwen transformer -18.6%）


def _reconfigure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# run 目录 / 进度
# ---------------------------------------------------------------------------
def ensure_run_dir(out, force=False):
    # ★ 子进程 cwd=仓库根（run_cmd 默认 REPO）⇒ 相对路径必须在编排器里先转绝对，
    #   否则 stage 会写到 D:\work\bitsandbytes-CPU\runs\... 而不是 video_mvp\runs\...
    out = os.path.abspath(out)
    if os.path.isdir(out) and os.listdir(out):
        if not force:
            raise SystemExit(f"[err] run 目录已存在且非空: {out}\n"
                             f"      加 --force 覆盖，或换一个 --out")
        for name in os.listdir(out):
            p = os.path.join(out, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path


def read_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def run_cmd(cmd, log_path, dry_run=False, cwd=REPO):
    """跑一个 stage 子进程（stdout/stderr → log）。返回退出码。"""
    log("  $ " + " ".join(cmd))
    if dry_run:
        log(f"    (dry-run: 未执行；log -> {log_path})")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    log(f"    -> exit {p.returncode}  ({time.time()-t0:.1f}s)  log={log_path}")
    if p.returncode != 0:
        tail = open(log_path, encoding="utf-8", errors="replace").read()[-1500:]
        log("    --- log tail ---\n" + tail)
    return p.returncode


# ---------------------------------------------------------------------------
# chunk 规划
# ---------------------------------------------------------------------------
def plan_chunks(mode, frames, chunk, overlap):
    """返回 (chunks, produced_frames)。produced = 实际会出多少帧。"""
    if mode == "independent":
        chunks = max(1, -(-frames // chunk))
        return chunks, chunks * chunk
    if mode == "sliding":
        stride = chunk - overlap
        if stride <= 0:
            raise SystemExit("[err] sliding 模式要求 chunk > overlap")
        chunks = 1 + max(0, -(-(frames - chunk) // stride))
        return chunks, chunk + (chunks - 1) * stride
    raise SystemExit(f"[err] 未知 mode: {mode}")


# ---------------------------------------------------------------------------
# stage: gen
# ---------------------------------------------------------------------------
def normalize_lora_path(lora, dry_run=False):
    """LoRA 路径规范化（**gen 与 smoke(full) 共用同一份逻辑**）。

    1) 相对路径 → 绝对路径：子进程 cwd=REPO，相对路径会去仓库根找（实测 FileNotFoundError）；
    2) 存在性断言：非 dry-run 时 `os.path.isfile` 必须成立，失败给清晰错误而不是让 safetensors 报。
    """
    if not lora:
        return ""
    p = os.path.abspath(lora)
    if not dry_run and not os.path.isfile(p):
        raise SystemExit(f"[err] LoRA 文件不存在: {p}\n"
                         f"      （若传的是训练 ckpt，请给 ckpt_last.pt 的绝对/相对路径）")
    return p


def build_gen_cmd(args, out, plan):
    common = [PY, os.path.join(E2E, "independent_chunks.py")]
    if args.mode == "sliding":
        common = [PY, os.path.join(E2E, "sliding_window.py")]
        cmd = common + [
            "--res", str(args.res), "--chunk", str(args.chunk), "--overlap", str(args.overlap),
            "--chunks", str(plan["chunks"]), "--steps", str(args.steps),
            "--carry-steps", str(args.carry_steps), "--carry-noise", str(args.carry_noise),
            "--threads", str(args.threads), "--seed", str(args.seed),
            "--outdir", out, "--scheduler", args.scheduler,
        ]
    else:
        cmd = common + [
            "--res", str(args.res), "--chunk", str(args.chunk), "--chunks", str(plan["chunks"]),
            "--steps", str(args.steps), "--threads", str(args.threads),
            "--seed", str(args.seed), "--outdir", out,
        ]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.neg:
        cmd += ["--neg", args.neg]
    if args.lora:
        # ★ gen 与 smoke(full) 共用 normalize_lora_path（绝对化 + isfile 断言）
        lora = normalize_lora_path(args.lora, dry_run=getattr(args, "dry_run", False))
        if lora.lower().endswith(".pt"):
            # PEFT ckpt → diffusers safetensors（convert_lora_ckpt.py 会打印配对自检）
            conv = os.path.join(out, "ckpt", "diffusers_lora.safetensors")
            os.makedirs(os.path.dirname(conv), exist_ok=True)
            rc = run_cmd([PY, os.path.join(REPO, "convert_lora_ckpt.py"),
                          "--ckpt", lora, "--out", conv],
                         os.path.join(out, "logs", "convert_lora.log"),
                         getattr(args, "dry_run", False))
            if rc != 0:
                raise SystemExit("[err] convert_lora_ckpt.py 失败（见 logs/convert_lora.log）")
            lora = conv
            if not getattr(args, "dry_run", False) and not os.path.isfile(lora):
                raise SystemExit(f"[err] convert 后 LoRA 仍不存在: {lora}")
        cmd += ["--lora", lora, "--lora-method", "manual"]
        cmd += ["--lora-scale", str(args.lora_scale)]
    return cmd


def stage_gen(args):
    if args.mode == "sliding" and args.lora:
        raise SystemExit("[err] sliding 模式的副本没有 LoRA 加载入口（原件也没有）——"
                         "要带自训 LoRA 生成请用 --mode independent（见 design doc §2.1）")
    out = ensure_run_dir(args.out, args.force)
    chunks, produced = plan_chunks(args.mode, args.frames, args.chunk, args.overlap)
    plan = dict(mode=args.mode, res=args.res, frames_requested=args.frames,
                chunk=args.chunk, overlap=args.overlap, chunks=chunks,
                produced_frames=produced, steps=args.steps, seed=args.seed,
                scheduler=args.scheduler)
    write_json(os.path.join(out, "plan.json"), plan)
    write_json(os.path.join(out, "config.json"), dict(
        schema=SCHEMA, stage="gen", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        python=sys.version.split()[0], threads=args.threads, argv=sys.argv, plan=plan,
        lora=args.lora, lora_scale=args.lora_scale, prompt=args.prompt, neg=args.neg,
        fps=args.fps, post=args.post))
    log(f"[gen] mode={args.mode} res={args.res} 目标 {args.frames} 帧 -> 实际 {produced} 帧 "
        f"（{chunks} chunks × {args.chunk}）  dry_run={args.dry_run}")
    cmd = build_gen_cmd(args, out, plan)
    rc = run_cmd(cmd, os.path.join(out, "logs", "gen.log"), args.dry_run)
    if rc != 0:
        raise SystemExit("[err] 生成 stage 失败（见 logs/gen.log）")
    if args.dry_run:
        log("[gen] dry-run 结束（未生成帧）")
        return 0
    frames = glob.glob(os.path.join(out, "frames", "f*.png"))
    log(f"[gen] 帧数={len(frames)}（计划 {produced}）")
    if len(frames) != produced:
        log(f"[warn] 实际帧数与计划不符：{len(frames)} != {produced}")
    # 后续 stage：默认自动接指标 + 后处理（同一 run 目录、顺序执行）
    if not args.no_metrics:
        rc = stage_metrics(_ns(out=out, chunk=args.chunk, overlap=args.overlap, dry_run=args.dry_run,
                               run_id=os.path.basename(out.rstrip("\\/"))))
        if rc != 0:
            return rc
    if args.post != "none" and not args.no_post:
        return stage_post(_ns(out=out, restorer=args.post, fps=args.fps, dry_run=args.dry_run,
                              run_id=os.path.basename(out.rstrip("\\/"))))
    return 0


# ---------------------------------------------------------------------------
# stage: metrics（统一 metrics.json）
# ---------------------------------------------------------------------------
def heuristic_verdict(metrics, mode):
    """按设计文档 §6 的 PASS 判据算 verdict（返回 dict(pass, reasons, warns)）。"""
    reasons, warns = [], []
    fr = metrics.get("frames") or {}
    if not fr.get("count"):
        reasons.append("没有帧")
    if fr.get("count") and not fr.get("all_finite", False):
        reasons.append("帧含非有限值")
    if fr.get("count") and (fr.get("std") or 0) <= 3.0:
        reasons.append(f"帧 std 过低 ({fr.get('std')}) ⇒ 近纯色/退化")
    if fr.get("count", 0) > 1 and (fr.get("frame_diff") or 0) <= 0.1:
        reasons.append(f"帧间差分过低 ({fr.get('frame_diff')}) ⇒ 静态")
    wh = metrics.get("white") or {}
    # E2E-PATCH[P7]: 只有"线性趋势成立（R²>0.5）"才算 whitening drift；否则只是逐帧方差。
    #   旧规则只看 slope>0.05，在 16 帧低分辨率噪声视频上会误报（实测 R²≈0.17–0.20）。
    if (wh.get("slope_per_frame") or 0) > 0.05 and (wh.get("r2") or 0) > 0.5:
        reasons.append(f"white% 漂移 slope={wh.get('slope_per_frame')} "
                       f"R2={wh.get('r2')} (>0.05/frame 且 R2>0.5)")
    elif (wh.get("max") or 0) > 10.0:
        warns.append(f"white% 峰值 {wh.get('max'):.1f}% >10%：高光过曝（疑似 hot VAE，"
                     f"设 ADIFF_VAE=D:\\work\\textmodel\\sd15_full\\vae 后复测）")
    de = metrics.get("detail") or {}
    if (de.get("drop_pct") or 0) > 40:
        reasons.append(f"细节衰减 {de.get('drop_pct'):.1f}% (>40%)")
    bd = metrics.get("boundary") or {}
    if mode == "sliding" and bd.get("ratio") is not None and bd["ratio"] > 1.5:
        reasons.append(f"滑窗 seam 可见 boundary/intra={bd['ratio']:.2f} (>1.5)")
    if mode == "independent" and bd.get("ratio") is not None:
        warns.append(f"independent 硬切 boundary/intra={bd['ratio']:.2f}（预期高，不算失败）")
    po = metrics.get("post") or {}
    if po:
        if not po.get("mp4_ok", False):
            warns.append("mp4 未写成（可能缺 av；已降级 gif）")
        if po.get("verify") and not po["verify"].get("ok", False):
            reasons.append("mp4 读回校验失败")
        # E2E-PATCH[P7b]: post_process.py 的冻结行为是 FSRCNN ×4 后 **area 抽回原尺寸**
        #   （细节恢复，不是放大）⇒ out_size 必须等于源帧尺寸；设计文档 §6 写的"512×512"是错的。
        if po.get("out_size") and fr.get("size") and list(po["out_size"]) != list(fr["size"]):
            warns.append(f"post out_size={po['out_size']} != 源尺寸 {fr['size']}"
                         f"（fsrcnn 应抽回原尺寸；如用 esrgan 放大则忽略此 warn）")
    return dict(pass_=not reasons, reasons=reasons, warns=warns)


def stage_metrics(args):
    out = os.path.abspath(args.out)
    frames_dir = os.path.join(out, "frames")
    if not args.dry_run and not glob.glob(os.path.join(frames_dir, "f*.png")):
        raise SystemExit(f"[err] 没有帧: {frames_dir}")
    tmp = os.path.join(out, "_metrics_tmp.json")
    chunk_metrics = os.path.join(out, "chunk_metrics.json")
    cmd = [PY, os.path.join(E2E, "temporal_metrics.py"), frames_dir,
           "--chunk", str(args.chunk), "--overlap", str(args.overlap),
           "--json-out", tmp, "--run-id", args.run_id or os.path.basename(out.rstrip("\\/"))]
    if os.path.exists(chunk_metrics):
        cmd += ["--extra-json", chunk_metrics]
    rc = run_cmd(cmd, os.path.join(out, "logs", "metrics.log"), args.dry_run)
    if rc != 0:
        raise SystemExit("[err] 指标 stage 失败（见 logs/metrics.log）")
    if args.dry_run:
        return 0
    m = read_json(tmp) or {}
    # 合并生成/后处理侧的元数据
    gen = read_json(os.path.join(out, "gen_meta.json"))
    chunk = read_json(chunk_metrics)
    post = read_json(os.path.join(out, "video.json")) or read_json(os.path.join(out, "post.json"))
    timing = dict(m.get("timing") or {})
    if gen:
        timing.update(gen_s=gen.get("seconds"), frames=gen.get("frames"),
                      mode=gen.get("mode"), chunk=gen.get("chunk"), overlap=gen.get("overlap"))
        m["gen"] = gen
    if chunk:
        m["chunks"] = chunk
    if post:
        m["post"] = dict(restorer=post.get("restorer"), scale=4,
                         out_size=post.get("out_size"), mp4=post.get("mp4"),
                         verify=post.get("verify"),
                         mp4_readback_frames=(post.get("verify") or {}).get("frames"),
                         mp4_ok=(post.get("mp4") is True or
                                 bool((post.get("verify") or {}).get("ok"))))
        timing["post_s"] = post.get("seconds")
    m["timing"] = timing
    m["schema"] = SCHEMA
    mode = (gen or {}).get("mode", "independent")
    v = heuristic_verdict(m, mode)
    m["verdict"] = dict(pass_=v["pass_"], reasons=v["reasons"], warns=v["warns"])
    write_json(os.path.join(out, "metrics.json"), m)
    os.remove(tmp)
    log(f"[metrics] verdict pass={v['pass_']} reasons={v['reasons']} warns={v['warns']}")
    return 0


# ---------------------------------------------------------------------------
# stage: post
# ---------------------------------------------------------------------------
def stage_post(args):
    out = os.path.abspath(args.out)
    if not args.dry_run and not glob.glob(os.path.join(out, "frames", "f*.png")):
        raise SystemExit(f"[err] 没有帧: {out}\\frames")
    cmd = [PY, os.path.join(E2E, "post_process.py"),
           "--src", os.path.join(out, "frames"), "--out", out,
           "--name", "video", "--fps", str(args.fps),
           "--restorer", args.restorer, "--verify", "--threads", str(THREADS_DEFAULT)]
    rc = run_cmd(cmd, os.path.join(out, "logs", "post.log"), args.dry_run)
    if rc != 0:
        raise SystemExit("[err] 后处理 stage 失败（见 logs/post.log）")
    if args.dry_run:
        return 0
    # 后处理结果并入 metrics.json 并重算 verdict
    vjson = read_json(os.path.join(out, "video.json"))
    m = read_json(os.path.join(out, "metrics.json"))
    if m is not None:
        if vjson:
            m["post"] = dict(restorer=vjson.get("restorer"), scale=4,
                             out_size=vjson.get("out_size"),
                             mp4=vjson.get("mp4"), verify=vjson.get("verify"),
                             mp4_readback_frames=(vjson.get("verify") or {}).get("frames"),
                             mp4_ok=bool(vjson.get("mp4")) or
                                    bool((vjson.get("verify") or {}).get("ok")))
        mode = ((m.get("gen") or {}).get("mode")
                or ((m.get("config") or {}).get("mode")) or "independent")
        v = heuristic_verdict(m, mode)
        m["verdict"] = dict(pass_=v["pass_"], reasons=v["reasons"], warns=v["warns"])
        write_json(os.path.join(out, "metrics.json"), m)
        log(f"[post] metrics 更新 pass={v['pass_']} reasons={v['reasons']}")
    return 0


# ---------------------------------------------------------------------------
# stage: train
# ---------------------------------------------------------------------------
def stage_train(args):
    out = ensure_run_dir(args.out, args.force)
    write_json(os.path.join(out, "config.json"), dict(
        schema=SCHEMA, stage="train", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        python=sys.version.split()[0], argv=sys.argv,
        data=args.data, stages=args.stages, rank=args.rank, scope=args.lora_scope,
        lr=args.lr, val_data=args.val_data, max_rss_gb=args.max_rss_gb))
    cmd = [PY, _tools_path("run_train_stages.py"),
           "--data", args.data, "--out", out, "--stages", args.stages,
           "--video_size", str(args.res), "--video_frames", str(args.frames),
           "--lora_scope", args.lora_scope, "--rank", str(args.rank), "--lr", str(args.lr),
           "--threads", str(args.threads), "--ckpt_every", str(args.ckpt_every),
           "--time_limit_s", str(args.time_limit_s), "--max_rss_gb", str(args.max_rss_gb)]
    if args.val_data:
        cmd += ["--val_data", args.val_data, "--val_every", str(args.val_every)]
    rc = run_cmd(cmd, os.path.join(out, "logs", "train.log"), args.dry_run)
    if rc != 0:
        raise SystemExit("[err] 训练 stage 失败（见 logs/train.log）")
    if args.dry_run:
        return 0
    ck = os.path.join(out, "ckpt_last.pt")
    log(f"[train] ckpt={ck} exists={os.path.exists(ck)}")
    if args.convert and os.path.exists(ck):
        conv = os.path.join(out, "ckpt", "diffusers_lora.safetensors")
        os.makedirs(os.path.dirname(conv), exist_ok=True)
        rc = run_cmd([PY, os.path.join(REPO, "convert_lora_ckpt.py"),
                      "--ckpt", ck, "--out", conv],
                     os.path.join(out, "logs", "convert_lora.log"), args.dry_run)
        if rc != 0:
            log("[warn] convert_lora_ckpt.py 失败（训练 ckpt 仍可用于 sample_check.py）")
    return 0


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def stage_report(args):
    out = os.path.abspath(args.out)
    cfg = read_json(os.path.join(out, "config.json")) or {}
    m = read_json(os.path.join(out, "metrics.json")) or {}
    plan = read_json(os.path.join(out, "plan.json")) or {}
    lines = [f"# video_e2e run: {os.path.basename(out.rstrip(chr(92)+'/'))}", ""]
    lines.append(f"- run 目录: `{out}`")
    if cfg:
        lines.append(f"- argv: `{' '.join(cfg.get('argv', []))}`")
    if plan:
        lines.append(f"- plan: mode={plan.get('mode')} res={plan.get('res')} "
                     f"chunk={plan.get('chunk')} chunks={plan.get('chunks')} "
                     f"steps={plan.get('steps')} produced_frames={plan.get('produced_frames')}")
    if m:
        fr = m.get("frames") or {}
        te = m.get("temporal") or {}
        wh = m.get("white") or {}
        de = m.get("detail") or {}
        bd = m.get("boundary") or {}
        po = m.get("post") or {}
        tm = m.get("timing") or {}
        lines += ["", "| 指标 | 值 |", "|---|---|",
                  f"| 帧数 / 尺寸 | {fr.get('count')} / {fr.get('size')} |",
                  f"| std / frame_diff | {fr.get('std')} / {fr.get('frame_diff')} |",
                  f"| MAD / warp_err / residual | {te.get('mad')} / {te.get('warp_err')} / {te.get('residual')} |",
                  f"| best_shift / has_global_motion | {te.get('best_shift')} / {te.get('has_global_motion')} |",
                  f"| white% slope / verdict | {wh.get('slope_per_frame')} / {wh.get('verdict')} |",
                  f"| detail drop% / verdict | {de.get('drop_pct')} / {de.get('verdict')} |",
                  f"| boundary ratio | {bd.get('ratio')} |",
                  f"| post restorer / out_size | {po.get('restorer')} / {po.get('out_size')} |",
                  f"| mp4 readback frames / ok | {po.get('mp4_readback_frames')} / {po.get('mp4_ok')} |",
                  f"| gen_s / metrics_s / post_s | {tm.get('gen_s')} / {tm.get('metrics_s')} / {tm.get('post_s')} |"]
        v = m.get("verdict") or {}
        lines += ["", f"**verdict: pass={v.get('pass_')}**  reasons={v.get('reasons')}  "
                      f"warns={v.get('warns')}", ""]
    md = "\n".join(lines) + "\n"
    p = args.md_out or os.path.join(out, "REPORT.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(md)
    log(md)
    log(f"[report] -> {p}")
    return 0


# ---------------------------------------------------------------------------
# stage: smoke（验收脚本，默认真跑；--dry-run 只打印）
# ---------------------------------------------------------------------------
SMOKE_EXPECT = {
    "infer": dict(time="~2.5–3 min", rss="≤7.6 GB（生成） / <3 GB（后处理）",
                  what="base AnimateDiff 128² 16 帧（independent 2×8），6 步 CFG7.5 → 指标 → FSRCNN → mp4"),
    "train": dict(time="~8.5 min（40 步，单段 ≤10min）", rss="≈6.6 GB",
                  what="data_win_S128_T8.pt（345+23 窗口）rank4 temporal 40 步 + held-out val → convert LoRA"),
    "full": dict(time="~10 min（训练 8.5min + 采样/指标/后处理各 <2min，分进程）",
                 rss="训练 6.6GB / 采样 ≤9.7GB / 后处理 <3GB（串行）",
                 what="train(20,40) → convert → 采样 + LoRA 生成 → 指标 → FSRCNN → mp4"),
}


def _ns(**kw):
    return argparse.Namespace(**kw)


def stage_smoke(args):
    prof = args.profile
    if prof not in SMOKE_EXPECT:
        raise SystemExit(f"[err] 未知 profile {prof}；可选 {list(SMOKE_EXPECT)}")
    info = SMOKE_EXPECT[prof]
    # ★ 绝对化：后续 lora/data 路径与子进程 cwd=REPO 相关（相对路径曾导致
    #   FileNotFoundError: runs\...\diffusers_lora.safetensors）
    root = os.path.abspath(args.root or os.path.join(RUNS, f"smoke_{prof}"))
    args.root = root
    if getattr(args, "data", ""):
        args.data = os.path.abspath(args.data)
    log(f"[smoke:{prof}] {info['what']}")
    log(f"[smoke:{prof}] 预期：耗时 {info['time']}；内存 {info['rss']}")
    log(f"[smoke:{prof}] run 根目录：{root}")
    stages = []
    reuse = bool(getattr(args, "from_ckpt", ""))
    if prof in ("train", "full") and not (prof == "full" and reuse):
        # ★ full + --from-ckpt：跳过训练段（训练实测 635s，C 要控回 ~5min）
        stages.append(("train", lambda: stage_train(_ns(
            out=os.path.join(root, "train"), force=args.force, data=args.data,
            stages="20,40", res=128, frames=8, lora_scope="temporal", rank=4, lr=1e-4,
            threads=4, ckpt_every=20, time_limit_s=540, max_rss_gb=9.5,
            val_data=args.data, val_every=20, convert=True, dry_run=args.dry_run))))
    if prof == "infer":
        stages.append(("gen+metrics+post", lambda: stage_gen(_ns(
            out=os.path.join(root, "infer"), force=args.force, mode="independent", res=128,
            frames=16, chunk=8, overlap=0, steps=6, carry_steps=3, carry_noise=400.0,
            threads=4, seed=1234, scheduler="dpm", prompt=None, neg=None, lora="",
            lora_scale=1.0, fps=8, post="fsrcnn", no_metrics=False, no_post=False,
            dry_run=args.dry_run))))
    if prof == "full":
        # ★ 默认复用已有 ckpt（--from-ckpt）：训练段实测 635s（40 步+convert），
        #   为了把 smoke C 控回 ~10min，允许跳过训练直接走"LoRA 生成→指标→后处理"。
        if args.from_ckpt:
            lora = normalize_lora_path(args.from_ckpt, dry_run=args.dry_run)
            log(f"[smoke:full] --from-ckpt 复用已有 ckpt，跳过训练段: {lora}")
        else:
            lora = os.path.join(root, "train", "ckpt", "diffusers_lora.safetensors")
        stages.append(("gen+metrics+post", lambda: stage_gen(_ns(
            out=os.path.join(root, "infer"), force=args.force, mode="independent", res=128,
            frames=16, chunk=8, overlap=0, steps=6, carry_steps=3, carry_noise=400.0,
            threads=4, seed=1234, scheduler="dpm", prompt=None, neg=None, lora=lora,
            lora_scale=1.0, fps=8, post="fsrcnn", no_metrics=False, no_post=False,
            dry_run=args.dry_run))))
    t0 = time.time()
    for name, fn in stages:
        log(f"[smoke:{prof}] ===== stage: {name} =====")
        rc = fn()
        if rc != 0:
            log(f"[smoke:{prof}] FAIL at {name} (rc={rc})")
            return rc
    log(f"[smoke:{prof}] 完成，用时 {time.time()-t0:.1f}s（dry-run={args.dry_run}）")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    _reconfigure_stdout()
    ap = argparse.ArgumentParser(
        prog="video_e2e.py",
        description="视频端到端管线编排器（Phase 1；只拼 subprocess，不 import torch）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
        p.add_argument("--force", action="store_true", help="覆盖非空 run 目录")

    # gen
    g = sub.add_parser("gen", help="生成（independent | sliding）→ frames → 指标 → 后处理")
    g.add_argument("--mode", choices=["independent", "sliding"], default="independent")
    g.add_argument("--res", type=int, default=128)
    g.add_argument("--frames", type=int, default=16, help="目标总帧数")
    g.add_argument("--chunk", type=int, default=8)
    g.add_argument("--overlap", type=int, default=0, help="sliding 模式重叠帧")
    g.add_argument("--steps", type=int, default=6)
    g.add_argument("--carry-steps", type=int, default=3)
    g.add_argument("--carry-noise", type=float, default=400.0)
    g.add_argument("--scheduler", default="dpm")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--lora", default="", help=".pt（PEFT ckpt，自动转换）或 .safetensors")
    g.add_argument("--lora-scale", type=float, default=1.0)
    g.add_argument("--prompt", default=None)
    g.add_argument("--neg", default=None)
    g.add_argument("--fps", type=int, default=8)
    g.add_argument("--post", default="fsrcnn", choices=["fsrcnn", "esrgan6b", "none"])
    g.add_argument("--no-post", action="store_true")
    g.add_argument("--no-metrics", action="store_true")
    g.add_argument("--out", default="")
    add_common(g)
    g.set_defaults(fn=stage_gen)

    # metrics
    mm = sub.add_parser("metrics", help="只重算指标 → 统一 metrics.json")
    mm.add_argument("--out", required=True)
    mm.add_argument("--run-id", default="")
    mm.add_argument("--chunk", type=int, default=8)
    mm.add_argument("--overlap", type=int, default=0)
    add_common(mm)
    mm.set_defaults(fn=stage_metrics)

    # post
    p = sub.add_parser("post", help="只做后处理（FSRCNN ×4 → gif/mp4 + 读回校验）")
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--restorer", default="fsrcnn", choices=["fsrcnn", "esrgan6b", "none"])
    p.add_argument("--fps", type=int, default=8)
    add_common(p)
    p.set_defaults(fn=stage_post)

    # train
    t = sub.add_parser("train", help="LoRA 训练（分段 resume）+ 可选 convert")
    t.add_argument("--data", required=True)
    t.add_argument("--val-data", default="")
    t.add_argument("--out", required=True)
    t.add_argument("--stages", default="20,40")
    t.add_argument("--res", type=int, default=128)
    t.add_argument("--frames", type=int, default=8)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--lora-scope", default="temporal")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--threads", type=int, default=5)
    t.add_argument("--ckpt-every", type=int, default=20)
    t.add_argument("--time-limit-s", type=float, default=540)
    t.add_argument("--max-rss-gb", type=float, default=9.5)
    t.add_argument("--val-every", type=int, default=20)
    t.add_argument("--convert", action="store_true", default=True)
    add_common(t)
    t.set_defaults(fn=stage_train)

    # smoke
    s = sub.add_parser("smoke", help="验收 smoke：infer / train / full")
    s.add_argument("--profile", choices=list(SMOKE_EXPECT), default="infer")
    s.add_argument("--root", default="")
    s.add_argument("--from-ckpt", default="",
                   help="full profile：复用已有 ckpt_last.pt（跳过训练段，C 控回 ~5min）")
    s.add_argument("--data", default=os.environ.get("VIDEO_E2E_DATA", ""),
                   help="训练数据缓存（外部 video_mvp/data/…；也可用 VIDEO_E2E_DATA 环境变量）")
    add_common(s)
    s.set_defaults(fn=stage_smoke)

    # report
    r = sub.add_parser("report", help="把 run 目录汇总成 markdown")
    r.add_argument("--out", required=True)
    r.add_argument("--md-out", default="")
    add_common(r)
    r.set_defaults(fn=stage_report)

    a = ap.parse_args()
    # gen 的默认 out 目录
    if a.cmd == "gen" and not a.out:
        a.out = os.path.join(RUNS, f"gen_{a.mode}_{a.res}_{a.frames}f_{int(time.time())}")
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
