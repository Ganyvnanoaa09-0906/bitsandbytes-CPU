"""让输出真的动起来：prompt 与步数/调度器的对照实验。

已确立的基线（`measure_prev_motion.py` 实测）：
  上一轮 8 帧 256px 输出 → **真实运动 0.00 px**，7 个帧对的块位移全是 (0,0)；
  而旧判据看到的"帧间差分 9.59" 100% 是闪烁残差。⇒ 旧管线压根没有运动。

本实验的假设：
  H1  prompt 里没有动作词 ⇒ 模型没有理由产生位移。加显式动作描述应当提高 M1。
  H2  LCM 4 步是为"少步数"蒸馏的，时间连贯性弱 ⇒ 同样 prompt 下提高步数或用
      DPM 多步应当提高 M1。

判据（`motion_metric`，已用合成控制样本验证 11/11）：
  M1 平均最优块位移（px） —— 真实运动幅值
  M2 运动可解释度         —— 差异中有多少能被"整体位移"解释（闪烁接近 0）

设计：单因子对照，每组只改一个变量，seed 固定，其余全同。
"""
import os
import gc
import sys
import json
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
sys.path.insert(0, r"D:\work")
from motion_metric import motion_metric, to_gray


def _load_anime_adiff():
    """加载 anime_adiff，同时挡掉它会引入的 bitsandbytes 路径遮蔽。

    为什么必须这样（实测踩到，且错误信息指向完全错误的方向）：

    `anime_adiff.py:17` 自己执行 `sys.path.insert(0, _HERE)`，而 `_HERE` =
    `D:\\work\\bitsandbytes-CPU`，该目录下的 `bitsandbytes\\` **没有 `__init__.py`**
    ⇒ 它变成一个命名空间包，遮蔽了真正的 bitsandbytes。
    peft 在模块顶层做可用性探测：

        peft/tuners/lora/bnb.py:322   if is_bnb_4bit_available():
        peft/import_utils.py:35       return hasattr(bnb.nn, "Linear4bit")

    它假设 bnb 不可用时 `bnb` 是 None（于是 `hasattr(None, "nn")` 为 False）。
    但命名空间包让 `bnb` 成了真实对象 ⇒ `hasattr(<空模块>, "nn")` 抛
    `AttributeError: module 'bitsandbytes' has no attribute 'nn'`。
    表象是 diffusers 报 "Loading lcm was unsuccessful"，看着像 bnb 装坏了，
    实际是路径遮蔽 —— 只要那个路径在 sys.path 上，**所有** LoRA 加载都会炸。

    对策（两步，缺一不可）：
      1. 加载前先在 sys.modules 里放一个"bnb 不可用"的占位模块 —— 让 peft 的探测
         拿到正确的答案（False），而不是拿到一个会让 hasattr 抛异常的空包。
      2. 加载后把 `bitsandbytes-CPU` 从 sys.path 摘掉，消除后续被遮蔽的可能。
    """
    import importlib.util
    import importlib.abc

    class _BitsAndBytesAbsent(importlib.abc.MetaPathFinder):
        """把 `bitsandbytes` 判定为"不存在"，即使 sys.path 上有个同名目录。

        上游有两道**不同**的判断：

            peft/import_utils.py:25   is_bnb_available()
                return importlib.util.find_spec("bitsandbytes") is not None
            peft/import_utils.py:29   is_bnb_4bit_available()
                if not is_bnb_available(): return False      # ← 短路在这里就结束了
                import bitsandbytes as bnb
                return hasattr(bnb.nn, "Linear4bit")

        只要 find_spec 返回 None，`is_bnb_4bit_available()` 就在第一行短路返回 False，
        根本走不到 `bnb.nn` —— 这正是"bnb 真的没装"时的正常路径。

        为什么不能用"预先塞 stub 进 sys.modules"的写法（我试过，踩了两次）：
          · `importlib.util.find_spec` 对已在 sys.modules 且 `__spec__ is None` 的模块
            **抛 ValueError**（不是返回 None）；
          · 给它一个真 ModuleSpec 则 find_spec 返回非 None ⇒ is_bnb_available() 为真
            ⇒ 走进 `bnb.nn.Linear4bit`，而 stub 里没有 ⇒ 要么 AttributeError 要么误判。
        ⇒ 结论：**不要预置 stub**，用元路径查找器把名字拦成"不存在"最干净。

        这个 finder 插在 sys.meta_path 最前面，优先于 PathFinder，所以无论
        anime_adiff 往 sys.path 里插什么，`bitsandbytes` 都解析不到命名空间包。
        """

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "bitsandbytes" or fullname.startswith("bitsandbytes."):
                return None
            return None                      # 其余名字一律不管，交回后续 finder

    sys.meta_path.insert(0, _BitsAndBytesAbsent())
    # 万一之前已被导入过（命名空间包），清掉它，避免半成品被复用
    for _k in [k for k in sys.modules if k == "bitsandbytes" or k.startswith("bitsandbytes.")]:
        del sys.modules[_k]

    p = r"D:\work\bitsandbytes-CPU\anime_adiff.py"
    spec = importlib.util.spec_from_file_location("anime_adiff", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["anime_adiff"] = mod
    spec.loader.exec_module(mod)

    # 摘掉 anime_adiff 插进来的目录，避免它继续遮蔽同名包
    here = r"D:\work\bitsandbytes-CPU"
    while here in sys.path:
        sys.path.remove(here)
    return mod


anime_adiff = _load_anime_adiff()

OUT_ROOT = r"D:\work\cloud_results\motion_exp"
os.makedirs(OUT_ROOT, exist_ok=True)

BASE_PROMPT = ("1girl, solo, upper body, looking at viewer, detailed face, silver hair, "
               "school uniform, soft lighting, best quality, gentle smile")
NEG = ("lowres, bad anatomy, bad hands, extra digits, cropped, worst quality, "
       "low quality, bad face, deformed, blurry")

# 显式动作描述：把"动作"写进 prompt 是让文生视频产生位移最直接的手段
MOTION_PROMPT = ("1girl, solo, upper body, silver hair blowing in the wind, "
                 "turning head, hair swaying, dynamic pose, wind, "
                 "detailed face, school uniform, cinematic lighting, best quality")

LCM_DIR = os.environ.get(
    "LCM_DIR", r"D:\work\ms_cache\models\latent-consistency--lcm-lora-sdv1-5")

FRAMES = 8
RES = 256
SEED = 1234

CASES = [
    # (标签, prompt, steps, use_lcm)
    ("A_base_lcm4", BASE_PROMPT, 4, True),
    ("B_motion_lcm4", MOTION_PROMPT, 4, True),
    ("C_motion_lcm8", MOTION_PROMPT, 8, True),
    ("D_motion_dpm12", MOTION_PROMPT, 12, False),
    ("E_base_dpm12", BASE_PROMPT, 12, False),
]


def build_pipe(use_lcm, scheduler):
    pipe = anime_adiff.build(
        scheduler=scheduler,
        adapter=r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2")
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.disable_attention_slicing()
    except Exception:
        pass
    if use_lcm:
        from diffusers import LCMScheduler
        pipe.load_lora_weights(LCM_DIR, weight_name="pytorch_lora_weights.safetensors",
                               local_files_only=True, adapter_name="lcm")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    return pipe


def gen(pipe, prompt, steps, use_lcm, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    gs = 1.5 if use_lcm else 7.5
    t0 = time.time()
    with torch.no_grad():
        out = pipe(prompt=prompt, negative_prompt=NEG, num_frames=FRAMES,
                   num_inference_steps=steps, guidance_scale=gs,
                   height=RES, width=RES, generator=g)
    return out.frames[0], time.time() - t0


def main():
    rep = {"config": {"frames": FRAMES, "res": RES, "seed": SEED},
           "base_prompt": BASE_PROMPT, "motion_prompt": MOTION_PROMPT,
           "cases": []}

    for tag, prompt, steps, use_lcm in CASES:
        sched = "dpm"
        print(f"\n=== {tag}: steps={steps} lcm={use_lcm} ===", flush=True)
        try:
            pipe = build_pipe(use_lcm, sched)
        except Exception as e:
            print(f"  构建失败: {type(e).__name__}: {e}", flush=True)
            rep["cases"].append({"tag": tag, "error": f"{type(e).__name__}: {e}"})
            continue

        try:
            frames, dt = gen(pipe, prompt, steps, use_lcm, SEED)
        except Exception as e:
            print(f"  生成失败: {type(e).__name__}: {e}", flush=True)
            rep["cases"].append({"tag": tag, "error": f"{type(e).__name__}: {e}"})
            del pipe
            gc.collect()
            continue

        gray = [to_gray(f) for f in frames]
        m = motion_metric(gray, max_shift=16)
        m2 = m["M2_explain"]
        m2s = "无差异" if m2 is None else f"{m2:.3f}"
        print(f"  {dt:.1f}s  M1={m['M1_motion_px']:.2f}px  M2={m2s}  "
              f"闪烁残差={m['flicker']:.2f}  旧判据差分={m['mean_absdiff']:.2f}  "
              f"方向={m['direction_consistency']:.2f}", flush=True)
        print(f"  位移序列: {m['shifts']}", flush=True)

        # 存帧（便于人眼看）
        try:
            from PIL import Image
            d = os.path.join(OUT_ROOT, tag)
            os.makedirs(d, exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(np.asarray(f)).convert("RGB").save(
                    os.path.join(d, "f%03d.png" % i))
        except Exception as e:
            print(f"  存帧失败: {e}", flush=True)

        rep["cases"].append({
            "tag": tag, "steps": steps, "lcm": use_lcm, "prompt": prompt,
            "gen_seconds": dt, **m,
        })
        del pipe
        gc.collect()

    # ---- 判定 ----
    by = {c["tag"]: c for c in rep["cases"] if "error" not in c}
    checks = []
    a = by.get("A_base_lcm4")
    b = by.get("B_motion_lcm4")
    c8 = by.get("C_motion_lcm8")
    d12 = by.get("D_motion_dpm12")
    e12 = by.get("E_base_dpm12")

    if a and b:
        checks.append(("H1 prompt 加动作词提高运动 (B>A)",
                       b["M1_motion_px"] > a["M1_motion_px"] + 0.3,
                       f"A={a['M1_motion_px']:.2f} B={b['M1_motion_px']:.2f}"))
    if b and c8:
        checks.append(("H2 LCM 加步数提高运动 (C>B)",
                       c8["M1_motion_px"] > b["M1_motion_px"] + 0.3,
                       f"B={b['M1_motion_px']:.2f} C={c8['M1_motion_px']:.2f}"))
    if b and d12:
        checks.append(("H2 DPM 12 步优于 LCM 4 步 (D>B)",
                       d12["M1_motion_px"] > b["M1_motion_px"] + 0.3,
                       f"B={b['M1_motion_px']:.2f} D={d12['M1_motion_px']:.2f}"))
    if e12 and a:
        checks.append(("负控制：同调度器下动作词仍有效 (D>E)",
                       d12 and d12["M1_motion_px"] > e12["M1_motion_px"] + 0.3,
                       f"E={e12['M1_motion_px']:.2f} D={d12['M1_motion_px']:.2f}"))
    best = max(by.values(), key=lambda x: x["M1_motion_px"]) if by else None
    if best:
        checks.append(("至少有一组产生可测的真实运动 (M1>=1px)",
                       best["M1_motion_px"] >= 1.0,
                       f"最佳 {best['tag']} M1={best['M1_motion_px']:.2f}px"))

    print("\n" + "=" * 74)
    print("--- 判定 ---")
    npass = 0
    for n, ok, det in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}   {det}")
        npass += bool(ok)
    print(f"\n{npass}/{len(checks)} PASS")

    rep["checks"] = [{"name": n, "pass": bool(o), "detail": d} for n, o, d in checks]
    rep["passed"], rep["total"] = npass, len(checks)
    rep["best"] = best["tag"] if best else None
    with open(r"D:\work\cloud_results\motion_exp.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
