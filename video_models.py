# -*- coding: utf-8 -*-
"""video_models.py — 视频模型加载的统一入口（可复用基础层）

为什么需要这一层
----------------
"视频模型的推理优化和训练支持"要能适配**任意** HF 视频模型，就不能把加载逻辑
散在各脚本里（现状：`profile_unet.py` 硬编码 SDXL 路径，`measure_video_cost2.py`
自己拼输入，`animatediff_train.py` 只认 `<dir>/unet + <dir>/motion_adapter`）。
本模块把「从磁盘拿到一个能跑 5D 前向的模型」收敛成两个函数：

    load_animatediff(unet_dir, adapter_dir) -> UNetMotionModel   # SD1.5 + MotionAdapter
    video_input(b, frames, res, ctx_dim, seed) -> (latent, t, ctx)

设计约束（实测踩出来的）
------------------------
1. **不要按模块名猜时序层**。本仓库实测：AnimateDiff 的时序模块命名里**没有**
   `temporal` 字样，但 `motion` 相关的模块有 639 个。按 `'temporal' in name` 去数
   会得到 0，从而误判"适配器没挂上"。要判断是否挂上，应该看**参数总量**与
   `motion` 命名模块数，而不是猜命名约定。
2. **`MotionAdapter.from_pretrained` 会对 config 里几个键发警告**
   （`motion_activation_fn` / `motion_attention_bias` / `motion_cross_attention_dim`），
   说它们"不被期望、将被忽略"。实测这些键**确实存在**于 diffusers 0.40.0 的
   `MotionAdapter.__init__` 之外，但 `from_pretrained` 依然能正确构造出
   block_out_channels 匹配的适配器。**警告不等于失效** —— 不要因为看到警告就去
   "修 config"，那会把一个能用的模型改坏。
3. 5D 前向的 latent 形状是 `(B, 4, F, H, W)`，**F 在最前面**（不是 (B,F,C,H,W)）。
   搞错这一维会得到形如 "size of tensor a (8192) must match b (1024)" 的报错，
   容易误判成模型坏了。

用法
----
    python video_models.py            # 自检：加载 + 2D/5D 前向 + 结构统计
"""
from __future__ import annotations

import gc
import os

import torch

DEFAULT_UNET = r'D:\work\textmodel\sd15_base\unet'
DEFAULT_ADAPTER = r'D:\work\textmodel\animatediff-motion-adapter-v1-5-2'


def load_animatediff(unet_dir: str = DEFAULT_UNET,
                     adapter_dir: str = DEFAULT_ADAPTER,
                     dtype=torch.float32):
    """SD1.5 UNet + AnimateDiff MotionAdapter -> UNetMotionModel（fp32，CPU）。"""
    from diffusers import MotionAdapter, UNet2DConditionModel
    from diffusers.models import UNetMotionModel

    u = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=dtype,
                                             local_files_only=True)
    a = MotionAdapter.from_pretrained(adapter_dir, torch_dtype=dtype,
                                      local_files_only=True)
    m = UNetMotionModel.from_unet2d(u, a)
    m.eval()
    del u, a
    gc.collect()
    return m


def structure(model) -> dict:
    """给出一份**可核对**的结构摘要，用于替代"按名字猜"。

    返回 motion 命名模块数、总参数量、以及各 block 的 channel 列表。
    """
    motion_named = sum(1 for n, _ in model.named_modules() if 'motion' in n.lower())
    temporal_named = sum(1 for n, _ in model.named_modules() if 'temporal' in n.lower())
    n_param = sum(p.numel() for p in model.parameters())
    return {
        'params': n_param,
        'params_M': n_param / 1e6,
        'motion_named_modules': motion_named,
        'temporal_named_modules': temporal_named,
        # 时序层是否真的注入：以 motion 命名模块数为准，不以 temporal 为准
        'motion_injected': motion_named > 0,
    }


def video_input(b: int = 1, frames: int = 8, res: int = 32, ctx_dim: int = 768,
                seed: int | None = 0, latent_channels: int = 4):
    """构造一次视频前向的输入。latent 是 (B, C, F, H, W) —— F 在 C 后面。

    ⚠️ **encoder_hidden_states 必须按帧数展开成 (B·F, 77, ctx_dim)**。
       原因：UNetMotionModel.forward 内部把 sample 从 (B,C,F,H,W) reshape 成
       (B·F, C, H, W) 再逐帧过 UNet（diffusers unet_motion_model.py:2012），
       所以 cross-attention 的 context 也必须已经是 B·F 行。若只给 B 行，
       会得到形如「size of tensor a (8192) must match b (1024)」的报错，
       而 **8192/1024 恰好等于帧数** —— 这个精确比值就是判据，不要把它误读成
       "模型坏了"或"时序层没注入"。（本仓库已在此误判过一次。）
    """
    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        latent = torch.randn(b, latent_channels, frames, res, res, generator=g)
        ctx1 = torch.randn(b, 77, ctx_dim, generator=g)
    else:
        latent = torch.randn(b, latent_channels, frames, res, res)
        ctx1 = torch.randn(b, 77, ctx_dim)
    ctx = ctx1.repeat(frames, 1, 1)          # (B,77,D) -> (B*F,77,D)
    t = torch.tensor([500] * b, dtype=torch.long)
    return latent, t, ctx


def forward_once(model, latent, t, ctx):
    with torch.no_grad():
        out = model(latent, t, encoder_hidden_states=ctx)
    return out.sample if hasattr(out, 'sample') else out


def _selftest() -> int:
    import time
    torch.set_num_threads(int(os.environ.get('THREADS', '6')))
    print('=' * 78)
    print('video_models.py 自检')
    print('=' * 78)
    t0 = time.perf_counter()
    m = load_animatediff()
    print('[load] %.1f s' % (time.perf_counter() - t0))
    st = structure(m)
    for k in ('params_M', 'motion_named_modules', 'temporal_named_modules', 'motion_injected'):
        print('  %-24s %s' % (k, st[k]))
    if not st['motion_injected']:
        print('  !! 时序层未注入 —— 这个模型不能用来做视频')
        return 1

    for frames, res in ((1, 32), (8, 32)):
        latent, t, ctx = video_input(1, frames, res)
        t0 = time.perf_counter()
        y = forward_once(m, latent, t, ctx)
        dt = time.perf_counter() - t0
        print('[fwd] frames=%-3d res=%d -> %s  %.2f s' % (frames, res, tuple(y.shape), dt))
        del latent, ctx, y
        gc.collect()
    print('OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(_selftest())
