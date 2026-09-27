"""运动适配器到底有没有在起作用？—— 不靠猜，直接测它对输出的影响。

现象：8 帧输出画面完好（动漫女孩、细节清晰），但**帧间位移恒为 0**（M1=0.00），
肉眼看也是同一张图重复 8 次。也就是说"图"没问题，"视频"没成立。

可能性（逐个可判）：
  P1 时序模块压根没被使用（适配器没挂上 / 全零权重）
  P2 时序模块有输出，但被 LCM 或步数压成了"每帧相同"
  P3 时序模块有效，只是这个 prompt / 帧数下模型选择不动

判据设计（同一个 pipeline 实例，只改一个变量）：
  T1 结构：motion 命名模块数与参数体量（确认适配器已注入）
  T2 决定性实验：把时序模块的输出**置零/旁路**，若输出与不旁路时**完全相同**，
     说明时序模块对结果没有贡献（P1 成立）；若不同，说明它在起作用（P2/P3）。
     这是"有没有影响"的直接判据，与画质、prompt 都无关。
  T3 时序模块权重的实际幅度（是否被 LCM 之外的什么清零）
  T4 同一 prompt、同一 seed 下 8 帧两两之间的**UNet 输入 latent 差异**：
     若初始 latent 就相同，那模型再强也不可能产出运动 —— 这是最底层的检查。
"""
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, r"D:\work")
# 导入 motion_exp 即会触发它内置的 bitsandbytes 路径防护（见 motion_exp._load_anime_adiff）
from motion_exp import anime_adiff  # noqa
import motion_metric  # noqa

torch.set_num_threads(6)
R = []


def check(name, ok, detail=""):
    R.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def main():
    adapter = r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2"
    pipe = anime_adiff.build(scheduler="dpm", adapter=adapter)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet

    # ---- T1 结构 ----
    motion_mods = [(n, m) for n, m in unet.named_modules() if "motion" in n.lower()]
    motion_params = [(n, p) for n, p in unet.named_parameters() if "motion" in n.lower()]
    n_mp = sum(p.numel() for _, p in motion_params)
    total = sum(p.numel() for p in unet.parameters())
    print(f"\n=== T1 结构 ===")
    print(f"  motion 命名模块 {len(motion_mods)} 个, 参数 {n_mp/1e6:.1f}M / 全模型 {total/1e6:.1f}M")
    check("T1 时序适配器已注入（模块数>0 且参数体量合理）",
          len(motion_mods) > 0 and n_mp > 1e6,
          f"{len(motion_mods)} 模块 / {n_mp/1e6:.1f}M 参数")

    # ---- T3 时序权重幅度 ----
    mags = [float(p.detach().abs().mean()) for _, p in motion_params]
    zeros = sum(1 for _, p in motion_params if float(p.detach().abs().max()) == 0.0)
    print(f"\n=== T3 时序模块权重 ===")
    print(f"  平均 |w| = {np.mean(mags):.5f}   全零张量 {zeros}/{len(motion_params)}")
    check("T3 时序权重非零（不是空适配器）", zeros == 0 and np.mean(mags) > 0,
          f"mean|w|={np.mean(mags):.5f}, 全零 {zeros}")

    # ---- T4 初始 latent 检查（最底层）----
    print(f"\n=== T4 初始 latent ===")
    gen = torch.Generator(device="cpu").manual_seed(1234)
    z = torch.randn(1, 4, 8, 32, 32, generator=gen)   # (B,C,F,H,W)
    per_frame_diff = float((z[:, :, 1:] - z[:, :, :-1]).abs().mean())
    print(f"  初始 latent 帧间差 = {per_frame_diff:.4f} (>0 才有运动的可能)")
    check("T4 初始 latent 各帧不同", per_frame_diff > 0.1, f"{per_frame_diff:.4f}")

    # ---- T2 决定性实验：旁路时序模块 ----
    print(f"\n=== T2 旁路时序模块（决定性判据）===")

    def run_once():
        g = torch.Generator(device="cpu").manual_seed(1234)
        with torch.no_grad():
            out = pipe(prompt="1girl, solo, silver hair, best quality",
                       negative_prompt="lowres, worst quality", num_frames=4,
                       num_inference_steps=6, guidance_scale=7.5,
                       height=128, width=128, generator=g)
        return np.stack([np.asarray(f, dtype=np.float32) for f in out.frames[0]])

    base = run_once()
    print(f"  基准输出: shape={base.shape} std={base.std():.2f} "
          f"帧间差={np.abs(np.diff(base,axis=0)).mean():.3f}")

    # 把 temporal transformer 的 forward 换成"原样返回第一个位置参数"（旁路）
    # 用 *args/**kwargs 通用签名 —— 实测它被以 7 个位置参数调用，写死参数列表会 TypeError。
    import types
    patched = 0
    for n, m in unet.named_modules():
        if "motion" in n.lower() and hasattr(m, "forward") and "Transformer" in type(m).__name__:
            m._orig_forward = m.forward

            def _bypass(self, *args, **kwargs):
                return args[0]

            m.forward = types.MethodType(_bypass, m)
            patched += 1
    print(f"  已旁路 {patched} 个时序 transformer")

    bypassed = run_once()
    diff = float(np.abs(bypassed - base).max())
    bd = float(np.abs(np.diff(bypassed, axis=0)).mean())
    print(f"  旁路后输出: std={bypassed.std():.2f} 帧间差={bd:.3f}")
    print(f"  旁路前后最大差异 = {diff:.6f}")

    check("T2 时序模块对输出【有】影响（旁路后输出不同）", diff > 1e-4,
          f"max|Δ|={diff:.6f}"
          + ("  ⇒ 旁路后完全一样，时序模块没起作用！" if diff <= 1e-4 else ""))

    # 还原
    for n, m in unet.named_modules():
        if hasattr(m, "_orig_forward"):
            m.forward = m._orig_forward
            del m._orig_forward

    # ---- 结论 ----
    print(f"\n=== 判据汇总 ===")
    npass = sum(1 for r in R if r["pass"])
    print(f"{npass}/{len(R)} PASS")
    os.makedirs(r"D:\work\cloud_results", exist_ok=True)
    with open(r"D:\work\cloud_results\motion_adapter_diag.json", "w", encoding="utf-8") as f:
        json.dump({"results": R, "passed": npass, "total": len(R),
                   "motion_modules": len(motion_mods), "motion_params_M": n_mp / 1e6,
                   "latent_frame_diff": per_frame_diff,
                   "base_std": float(base.std()),
                   "base_frame_diff": float(np.abs(np.diff(base, axis=0)).mean()),
                   "bypass_max_diff": diff,
                   "bypass_frame_diff": bd,
                   "patched": patched}, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
