# -*- coding: utf-8 -*-
"""loop_transformer.py — 循环 Transformer 的三个组件

设计依据（实测 + 论文）:
  · 我们的 T=64 AR 打满了 8×8 分词器天花板（19.0 dB）⇒ 容量够
  · 但 T=256 的 AR 差 16×16 天花板很远 ⇒ 【任务难 4 倍，容量不够】
    （信息量: 64×10=640 bit vs 256×10=2560 bit）
  · 而数据只有 7.45M token ⇒ 【加参数有过拟合风险】
  ⇒ 【用深度换容量】是正解

三个组件（对应讨论的①②③）:

  ① LoopStepEncoding —— 循环步数编码
     没有它，共享层只能学一个"不动点迭代"，无法表达"第1轮粗筛、第3轮细化"
     可学习表 + 正弦外推（后者解决训练 L=3、推理 L=8 的外推问题）

  ② 随机循环轮数
     训练时 L ~ U{1..L_max}（按 batch 统一，避免 batch 内掩码复杂度）
     推理时可弹性选 L
     ⚠️ KV cache 一致性：循环块用 append=False 复用 cache，
        所以 L 变化【不影响 cache】—— 这是当前实现天然支持的

  ③ GatedCrossIteration —— 跨轮门控残差
     循环的瓶颈是【信息瓶颈】：每轮都过同一组层，信息被反复压缩
     跨轮残差让每轮都能"看到"循环入口的表示，用门控决定看多少
     成本 ≈ 0（每轮一个标量门），比"跨轮注意力"（O(L²T²)）便宜两个数量级

参考:
  · mHC (arXiv 2512.24880): 双随机约束的流间混合 —— 更重，留作备选
  · Attention Residuals: 跨层 softmax attention —— 循环展开后等价于跨轮
  · 用户建议: 33M 规模先上轻量的门控，震荡再上 mHC 的双随机约束
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
import math
import torch
import torch.nn


class LoopStepEncoding(nn.Module):
    """循环步数编码：让共享层知道自己在第几轮。

    可学习表处理训练范围内的步数（max_steps），
    超出范围时回退到固定的正弦编码 ⇒ 支持推理时外推到更大的 L。
    """

    def __init__(self, d_model, max_steps=8):
        super().__init__()
        self.d = d_model
        self.max_steps = max_steps
        self.table = nn.Embedding(max_steps, d_model)
        nn.init.zeros_(self.table.weight)     # 初始为 0 ⇒ 等价于没加编码
        # 正弦部分固定（不可学习），用于外推
        half = d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs, persistent=False)

    def _sinusoidal(self, step, device, dtype):
        ang = float(step) * self.freqs.to(device=device, dtype=dtype)   # (half,)
        return torch.cat([ang.sin(), ang.cos()], dim=0)                 # (d,)

    def forward(self, step, device, dtype):
        """返回 (d,) 的步数编码向量。step 从 0 开始（第 0 轮 = 首次进入）。"""
        if step < self.max_steps:
            v = self.table(torch.tensor(step, device=device))
            return v.to(dtype)
        return self._sinusoidal(step, device, dtype)


class GatedCrossIteration(nn.Module):
    """跨轮门控残差：让每轮都能看到循环入口的表示。

    形式：
        x_t = block(x_{t-1}, step=t)
        x_t = x_t + g_t ⊙ x_entry
    其中 g_t 是由 [x_t; x_entry] 算出的门（每通道），
    初始化为接近 0（不干扰原有行为）。

    为什么需要它：
        循环块复用同一份 KV cache，所以每轮 attend 的是【同样的上下文】。
        没有跨轮连接时，多轮迭代只在"深度"方向堆叠，信息会被反复压缩。
        跨轮残差开了一条"旁路"，直接对应 Attention Residuals 想解决的问题，
        但成本是每轮一个门（O(1)），而不是注意力（O(L²T²)）。
    """

    def __init__(self, d_model, max_steps=8, init_bias=-4.0):
        super().__init__()
        self.d = d_model
        self.max_steps = max_steps
        # per-step 的通道门（共享给所有 token）
        self.gate = nn.Parameter(torch.full((max_steps, d_model), float(init_bias)))
        # 用 [x_t - x_entry] 的内容再调一次（内容相关门）
        self.dyn = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.dyn.weight)

    def forward(self, x, x_entry, step):
        """x: (B,T,d) 本轮的 block 输出；x_entry: (B,T,d) 循环入口表示"""
        if step >= self.max_steps:
            g = torch.zeros_like(x)
        else:
            g = torch.sigmoid(self.gate[step])                      # (d,)
            g = g.view(1, 1, -1)
        # 内容相关标量门（初始为 0 ⇒ sigmoid=0.5，但被上面的 bias 压制）
        s = torch.sigmoid(self.dyn(x - x_entry))                    # (B,T,1)
        return x + g * s * x_entry


class LoopConfig:
    """循环配置（从 argparse 构造）"""

    def __init__(self, loop_start=4, loop_end=7, loop_L=1, loop_L_max=1,
                 loop_random=False, loop_p=None, step_enc=True, cross_res=True,
                 loop_cache='shared'):
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.loop_L = loop_L              # 固定轮数（loop_random=False 时用）
        self.loop_L_max = loop_L_max      # 随机采样上界
        self.loop_random = loop_random
        self.loop_p = loop_p          # None=均匀；否则长度=L_max 的 p(L=1..L_max)
        self.step_enc = step_enc
        self.cross_res = cross_res
        # 'shared': 整个循环共用一份 KV，只有第 0 轮写入（历史行为）
        # 'perloop': 每一轮各自一份 KV（训练时的行为，见 run_blocks_looped 的注释）
        self.loop_cache = loop_cache

    def sample_L(self):
        """训练时采样本轮循环轮数。按 batch 统一（不做元素级掩码）。"""
        if not self.loop_random:
            return max(1, self.loop_L)
        if self.loop_p:
            r = float(torch.rand(1).item())
            acc = 0.0
            for i, pi in enumerate(self.loop_p):
                acc += float(pi)
                if r < acc:
                    return i + 1
            return len(self.loop_p)
        return int(torch.randint(1, self.loop_L_max + 1, (1,)).item())


def make_loop_modules(cfg_model, loop_cfg):
    """按配置创建循环相关的模块（注册进模型，方便 state_dict 保存）"""
    mods = nn.ModuleDict()
    d = cfg_model.d_model
    mx = max(2, loop_cfg.loop_L_max + 1)
    if loop_cfg.step_enc:
        mods['step_enc'] = LoopStepEncoding(d, max_steps=mx)
    if loop_cfg.cross_res:
        mods['cross_res'] = GatedCrossIteration(d, max_steps=mx)
    return mods


def run_blocks_looped(blocks, x, loop_idx, loop_cfg, loop_mods,
                      pooled, ctx, caches, keys=None,
                      L_override=None):
    """带循环的 block 前向。

    与原始 run_blocks 的区别：
      · 循环轮数可变（训练时随机，推理时可指定）
      · 每轮注入步数编码
      · 每轮做跨轮门控残差

    ⚠️ 与 KV cache 的关系：
        循环块在首次进入时 append=True（写入 cache），后续轮 append=False（复用 cache）。
        所以 L 变化【不影响 cache 的正确性】。

    ⚠️⚠️ 但"复用"这件事本身与训练不一致，这是 2026-10-02 发现的：
        训练路径从不传 caches（forward_logits / forward_masked / forward_hier 全是 None），
        所以每一轮都是【用当轮隐状态重算 K/V】再 attend —— 第 t 轮 attend 的是 loop-t 的 K/V。
        而采样时 append=(t==0) 意味着第 t>0 轮 attend 的是【第 0 轮写下的 K/V】。
        ⇒ 第 1 轮以后，训练与推理看到的上下文根本不是同一份。

        两篇 2026 年的工作独立测过这个取舍，都选了"每轮独立"：
          · Nanbeige4.2-3B (arXiv:2607.22083)：两遍共享 KV 让 cache 减半，但效果变差；
          · Looped Latent Attention (arXiv:2607.15456)：循环 K/V 沿循环轴低秩，
            但"不能安全地塌成单一状态"，per-head 编解码打败 final-loop reuse。

        cache 传 list 时即走 'perloop'：第 t 轮用第 t 份 cache，且每轮都 append=True，
        与训练的语义对齐。轮数此时取 len(cache)，保证每个槽位都被写过（否则错位）。
    """
    B, T, _ = x.shape
    perloop = caches is not None and len(caches) > 0 and isinstance(caches[0], list)
    if perloop:
        # 槽位数就是轮数：L 随 cache 分配走，避免有的槽位从没被写过
        L = len(caches[0]) if L_override is None else L_override
    else:
        L = L_override if L_override is not None else loop_cfg.sample_L()
    rnd = 0

    for i, blk in enumerate(blocks):
        if i not in loop_idx:
            x = blk(x, pooled=pooled, ctx=ctx, cache=caches[i] if caches else None,
                    append=True, keys=keys)
            continue

        # ---- 进入循环块 ----
        x_entry = x                      # 循环入口表示（跨轮残差用）
        cache = caches[i] if caches else None
        slots = cache if isinstance(cache, list) else None
        for t in range(L):
            # 共享模式：只有首轮写入；每轮独立模式：每轮写自己那一份
            slot = slots[t] if slots is not None else cache
            append = True if slots is not None else (t == 0)
            xin = x
            if loop_cfg.step_enc and t > 0:
                xin = xin + loop_mods['step_enc'](t, x.device, x.dtype).view(1, 1, -1)
            x = blk(xin, pooled=pooled, ctx=ctx, cache=slot,
                    append=append, keys=keys)
            if loop_cfg.cross_res and t > 0:
                x = loop_mods['cross_res'](x, x_entry, t)
            rnd += 1
    return x
