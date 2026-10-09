# -*- coding: utf-8 -*-
"""small_image_model_v2.py - AR 图像模型架构改进版（对照用户的评审清单）。

落实项
------
P0-1 真实条件注入（v1 训练时 text 全为 0，模型其实是无条件）
     cond_mode: 'none' | 'add' | 'adaln'(每层 scale/shift) | 'cross'(对文本 token 序列做 cross-attn)
     cond_dropout: 训练时随机丢条件，为 CFG 铺路
P0-2 2D 位置编码：可学习 row/col 嵌入（v1 是 1D 可学习 + 光栅序），
     位置由 (r,c) 直接查表，因此天然支持非光栅 token 顺序
P0-3 Gen/Think 头因子化：vocab = n_cluster x (vocab/n_cluster)，输出计算量降约 16 倍
P0-4 KV Cache：增量解码，每步只前向 1 个 token（v1 每步重算全序列）
P1-5 RMSNorm 取代 LayerNorm
P1-6 SwiGLU FFN（同参数量下自动折算 hidden）
P1-7 QK Norm
P1-8 GQA/MQA（n_kv_head 可配，KV Cache 随之缩小）
P1-9 LayerScale（10 层小模型更稳）

用法
----
    from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    m = SmallARImageModelV2(SmallImageConfigV2())
    logits = m.forward_logits(tokens, text_cond)     # 训练：一次拿全序列 logits
    toks   = m.generate(text_cond, top_k=100)        # 推理：走 KV Cache
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 融合 RMSNorm 开关。默认开；CPU_FORGE_RMSNORM=legacy 退回逐算子旧实现。
# 注意：这个模块是被 train_ar_v2.py / 采样脚本共用的，改这里【全局生效】——
# 所以数值等价性必须先过 check_rmsnorm_equiv.py，且保留退回路径。
_RMS_FUSED = (os.environ.get('CPU_FORGE_RMSNORM', 'fused').lower() != 'legacy') \
             and hasattr(F, 'rms_norm')


# ------------------------------------------------------------------ 基础件
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """★ 融合 RMSNorm（2026-09-24 改）

        原实现是一句话：
            self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        它在 autograd 里展开成 6 个独立 kernel：pow -> mean -> add -> rsqrt -> mul -> mul。
        每个 kernel 都要把整个张量读一遍、写一遍，是纯内存带宽开销。

        profile_ar_ops.py 的算子剖析（v4_bos/ckpt_13000, B=8 T=256, 10 层）显示：
            aten::mm                       46.9%
            mul 15.7 + sum 5.3 + sqrt 1.9 + div 1.6 + pow 1.5 = 26.0%   <- 就是这里
            SDPA 5.6 / bnb 优化器 4.8 / copy_ 3.6 / cat 1.0

        换成 F.rms_norm 后是单个 kernel。数值等价性已由 check_rmsnorm_equiv.py 验证：
          相对峰值误差 ~1e-7，grad_x 2.5e-7 / grad_w 9.9e-8，
          全零 / 全常数 / 1e-20 / 单个 1e30 等退化输入行为一致。
        设 CPU_FORGE_RMSNORM=legacy 可退回旧实现，用于 A/B 对照。
        """
        if _RMS_FUSED:
            return F.rms_norm(x, (x.shape[-1],), self.w, self.eps)
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        hidden = max(8, (int(ffn_dim * 2 / 3) + 7) // 8 * 8)
        self.w12 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


def make_token_order(grid, order='raster'):
    """返回长度 grid*grid 的排列：新序第 i 位对应原光栅序的 perm[i]。"""
    n = grid * grid
    if order == 'raster':
        return list(range(n))
    if order == 'zorder':
        bits = max(1, int(math.log2(grid)))

        def morton(r, c):
            v = 0
            for b in range(bits):
                v |= ((r >> b) & 1) << (2 * b + 1)
                v |= ((c >> b) & 1) << (2 * b)
            return v
        pairs = sorted(((r, c) for r in range(grid) for c in range(grid)),
                       key=lambda rc: morton(*rc))
        return [r * grid + c for r, c in pairs]
    raise ValueError('未知 token_order: %s' % order)


def build_cluster_map(codebook_weight, n_cluster, iters=25, seed=0):
    """在 VQ 码本上做 k-means，返回**均衡**的簇划分。

    必须均衡：因子化头按 vocab//n_cluster 输出 code logits，
    若某簇塞了 96 个码而 per 只有 32，两边形状就对不上。
    做法：k-means 求中心后，按「最近中心距离」排序，贪心地把码分配给它最靠近
    且仍有空位的簇，保证每簇恰好 vocab//n_cluster 个。

    返回 (cluster_of_code, offset_of_code, per)。
    """
    W = codebook_weight.detach().float()
    V = W.shape[0]
    assert V % n_cluster == 0, 'vocab 必须能被 n_cluster 整除'
    per = V // n_cluster
    g = torch.Generator().manual_seed(seed)
    C = W[torch.randperm(V, generator=g)[:n_cluster]].clone()
    for _ in range(iters):
        a = torch.cdist(W, C).argmin(dim=1)
        for k in range(n_cluster):
            m = a == k
            if m.any():
                C[k] = W[m].mean(dim=0)

    D = torch.cdist(W, C)                       # V x K
    order = D.min(dim=1).values.argsort(descending=True)   # 最"孤立"的先分配
    cap = torch.full((n_cluster,), per, dtype=torch.long)
    cluster_of = torch.full((V,), -1, dtype=torch.long)
    for i in order.tolist():
        for k in D[i].argsort().tolist():
            if cap[k] > 0:
                cluster_of[i] = k
                cap[k] -= 1
                break
    # 簇内连续编号
    offset_of = torch.zeros(V, dtype=torch.long)
    for k in range(n_cluster):
        m = (cluster_of == k).nonzero(as_tuple=True)[0]
        offset_of[m] = torch.arange(len(m))
    return cluster_of, offset_of, per


# ------------------------------------------------------------------ 注意力
def _make_attn(cfg):
    """Build the attention module selected by cfg.attn_kind.

    'mha' (default) is the original SelfAttention, so nothing changes unless a
    config asks for it. 'gdn' swaps in kda_model.KDAttention, a chunked gated
    delta-rule linear attention whose forward signature matches -- (B,T,C) to
    (B,T,C) -- and whose cost is O(T*C*D + T*D^2) rather than O(T^2*D).

    Measured with video_attn_vs_gdn.py at identical shapes in one process, the
    speed ratio over attention grows with T: 5.24x at 1152, 7.24x at 2304, 9.37x
    at 4608, 15.20x at 9216, with GDN itself scaling exactly linearly.
    """
    kind = str(getattr(cfg, 'attn_kind', 'mha') or 'mha').lower()
    if kind in ('mha', 'self', 'attention'):
        return SelfAttention(cfg)
    if kind in ('gdn', 'kda', 'linear'):
        try:
            from kda_model import KDAttention
        except Exception as e:                       # pragma: no cover
            raise RuntimeError(
                "attn_kind=%r needs kda_model.py: %s" % (kind, e))
        d, nh = cfg.d_model, cfg.n_head
        kda = KDAttention(d, nh,
                          chunk_size=int(getattr(cfg, 'chunk_size', 64) or 64),
                          use_qk_norm=bool(getattr(cfg, 'use_qk_norm', True)))

        class _KDAAdapter(nn.Module):
            """Match SelfAttention's call signature.

            SelfAttention.forward is (x, cache=None, append=True) and the block
            passes both, while KDAttention.forward only takes x. GDN keeps no
            per-token cache -- its whole point is a fixed-size recurrent state --
            so the extra arguments are accepted and ignored rather than threaded
            through. Without this the block raises at the first forward pass.
            """

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x, cache=None, append=True):
                return self.inner(x)

        return _KDAAdapter(kda)
    raise ValueError('unknown attn_kind %r' % kind)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, nh = cfg.d_model, cfg.n_head
        assert d % nh == 0 and nh % cfg.n_kv_head == 0
        self.nh, self.n_kv = nh, cfg.n_kv_head
        self._mask_mode = getattr(cfg, 'mask_mode', 'causal')
        self._patch_nums = tuple(getattr(cfg, 'patch_nums', ()) or ())
        # 多尺度注意力（mask_mode='msa'）的尺度定义。见 _msa 的注释：
        # 组大小按【网格】定，而不是照抄 CSA/HCA 的 ×4/×128。
        self._grid = int(getattr(cfg, 'grid', 16) or 16)
        _g = getattr(cfg, 'msa_groups', '') or ''
        if _g:
            self._msa_groups = tuple(int(x) for x in str(_g).replace(' ', '').split(',') if x)
        else:
            # 行摘要 + 四行摘要。不用 ×T：全局摘要要整幅图生成完才存在，因果下只有
            # 最后一个 token 用得上（真正的全局得靠 running state，那是 KDA 的事）。
            self._msa_groups = (self._grid, 4 * self._grid)
        self._msa_local = int(getattr(cfg, 'msa_local', 0) or 2 * self._grid)
        self.hd = d // nh
        # GQA 正确做法：KV 头维度与 Q 头相同，总 KV 维度 = n_kv_head * hd。
        # 若写成 d//n_kv_head，KV 维度恒为 d，缓存一点都不会变小。
        self.kvh = self.hd
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.kvh)
        # QKV 融合：三个小 GEMM 合成一个。实测（bench_qkv.py）在 batch*T=512~2048
        # 时能快 16~26% —— 少了 2/3 的 kernel 启动，且 GEMM 的 N 维度 512/256/256 -> 1024。
        # 默认关闭以保持和旧 checkpoint 的兼容（state_dict 键名不同）。
        self.fuse_qkv = bool(getattr(cfg, 'fuse_qkv', False))
        if self.fuse_qkv:
            self.qkv = nn.Linear(d, nh * self.hd + 2 * cfg.n_kv_head * self.kvh, bias=False)
        else:
            # 只在非融合时创建 —— 否则会多出 5M 无用参数，白白增加优化器开销
            self.q = nn.Linear(d, nh * self.hd, bias=False)
            self.k = nn.Linear(d, cfg.n_kv_head * self.kvh, bias=False)
            self.v = nn.Linear(d, cfg.n_kv_head * self.kvh, bias=False)
        self.o = nn.Linear(nh * self.hd, d, bias=False)

    def _qkv(self, x, B, T):
        if self.fuse_qkv:
            qq, kk, vv = self.qkv(x).split(
                [self.nh * self.hd, self.n_kv * self.kvh, self.n_kv * self.kvh], dim=-1)
        else:
            qq, kk, vv = self.q(x), self.k(x), self.v(x)
        return (self.q_norm(qq.view(B, T, self.nh, self.hd)).transpose(1, 2),
                self.k_norm(kk.view(B, T, self.n_kv, self.kvh)).transpose(1, 2),
                vv.view(B, T, self.n_kv, self.kvh).transpose(1, 2))

    def forward(self, x, cache=None, append=True):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, B, T)

        # GQA：把 KV 从 n_kv 头展开到 nh 头。
        # 【关键优化】缓存里存【已展开】的版本，每个增量步只展开新来的那个位置。
        # 原来每步都对整个 cache 做 repeat_interleave —— T=256、10 层时是 25 ms/步，
        # 占了整个生成步骤的 44%（实测 batch=16）。
        rep = self.nh // self.n_kv

        def expand(t):
            return t.repeat_interleave(rep, dim=1) if rep > 1 else t

        if cache is not None:
            cap = cache.get('cap')
            if cap is not None:
                # 【预分配路径】cache 里是一块 (B, nh, cap, hd) 的缓冲，
                # 每步只写切片，避免 aten::cat —— 实测 cat 占生成总时间的 14.5%。
                # cap/n 由 generate() 建 cache 时给出。
                n = cache.get('n', 0)
                if cache.get('k') is None:
                    cache['k'] = torch.empty(B, self.nh, cap, self.hd,
                                             dtype=k.dtype, device=k.device)
                    cache['v'] = torch.empty(B, self.nh, cap, self.hd,
                                             dtype=k.dtype, device=k.device)
                    cache['n'] = 0
                    n = 0
                if append:
                    if n + T > cap:
                        raise RuntimeError('KV cache 溢出: %d+%d > %d' % (n, T, cap))
                    cache['k'][:, :, n:n + T, :] = expand(k)
                    cache['v'][:, :, n:n + T, :] = expand(v)
                    cache['n'] = n + T
                    kk = cache['k'][:, :, :n + T, :]
                    vv = cache['v'][:, :, :n + T, :]
                else:
                    kk = cache['k'][:, :, :n, :]
                    vv = cache['v'][:, :, :n, :]
                causal = (T > 1)
            else:
                # 【旧路径】cat 累加（向后兼容，train/不带 cap 的调用走这里）
                if cache.get('k') is not None:
                    if append:
                        k = torch.cat([cache['k'], expand(k)], dim=2)
                        v = torch.cat([cache['v'], expand(v)], dim=2)
                    else:
                        k, v = cache['k'], cache['v']
                else:
                    k, v = expand(k), expand(v)
                if append:
                    cache['k'], cache['v'] = k, v
                kk, vv = k, v
                causal = (T > 1)
        else:
            causal = True
            kk, vv = expand(k), expand(v)

        # ---- VAR 风格 block-causal mask ----
        # 动机（2026-09-18 实测）：光栅序逐 token causal 下，T=256 的 AR
        #   ① 生成 token 相邻同率 42~54%（真实 4.47%）⇒ token 坍塌
        #   ② 逐位置 top-1 越后越差（11.5% → 5.7%）⇒ 长程依赖失效
        # 而字节 VAR（arXiv 2404.02905, NeurIPS 2024 Best Paper）用
        #   "next-scale prediction + 尺度内双向注意力"把 FID 从 18.65 降到 1.73。
        # 本实现借其核心思想：token 按 Z-order 排列（粗到细），
        # 掩码规则等价于 VAR 的 lvl(i) >= lvl(j) ⇒ 块内双向、块间因果。
        # ⇒ 自回归步数 256 → log4(256)+1 = 5，且块内保留完整 2D 结构。
        mask_mode = getattr(self, '_mask_mode', 'causal')
        if mask_mode == 'full' and T > 1:
            # GRN 风格：全双向（无因果）。每个位置都能看到所有其他位置。
            # 论文依据：GRN (arXiv 2604.13030) 用 global refinement，
            # 所有位置平等、可反复修改，从根本上绕开因果序对 2D 结构的破坏。
            y = F.scaled_dot_product_attention(q, kk, vv)
            return self.o(y.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd))
        if causal and mask_mode == 'msblock' and T > 1:
            blk = self._ms_block_mask(T, q.device)
            if blk is not None:
                y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=blk)
                return self.o(y.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd))
        # ⚠️ 守卫必须在 T>1 判断【之前】：采样时 T=1，若放在后面，带 cache 的单步调用会
        # 绕开整个 msa 分支、静默走普通因果注意力 —— 训练与推理两种注意力，还不报错。
        # （自测第一次就是这么挂的：msa_selftest.py 第 3 条 FAIL。）
        if mask_mode == 'msa' and cache is not None:
            raise RuntimeError(
                "mask_mode='msa' 不能走增量 KV cache：压缩摘要要等一组 token 全部到齐"
                "才算得出来。请用逐步全量重算的采样器（i5build/msa_sample.py），"
                "不要用 generate()。")
        if causal and mask_mode == 'msa' and T > 1:
            y = self._msa(q, kk, vv, T)
            return self.o(y.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd))
        if causal and mask_mode == 'block' and T > 1:
            blk = self._block_mask(T, kk.shape[2], q.device)
            if blk is not None:
                y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=blk)
                return self.o(y.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd))
        y = F.scaled_dot_product_attention(q, kk, vv, is_causal=causal)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd))

    def _msa(self, q, kk, vv, T):
        """多尺度注意力：局部窗口 + 行摘要(×grid) + 四行摘要(×4·grid)。

        思路借自 CSA/HCA（DeepSeek-V4，arXiv:2606.19348）的"沿序列轴压缩"：
        把一段历史压成一条 KV，再让查询去注意压缩后的条目。**但组大小必须按二维网格
        重定**：光栅序下一行 = grid 个 token，所以 ×grid 才是"行摘要"；照抄 ×4 得到的是
        "四分之三行"，不对应任何空间结构。

        因果性：查询 p 只能看【已经完整结束】的组，即组末 (i+1)·g − 1 < p。
        因此 ×T 的全局摘要在自回归里不可用（要整幅图都生成完），真正的全局信息只能靠
        一个 running state —— 那是 KDA/线性注意力要做的事，不是掩码能补的。

        摘要用组内均值。局部窗口保留未压缩 KV，与两个压缩尺度拼在一次 softmax 里。
        """
        B, H, S, hd = kk.shape
        dev = q.device
        p = torch.arange(T, device=dev)
        # 局部：因果 + 只看最近 _msa_local 个位置
        mask = (p[None, :] <= p[:, None]) & (p[None, :] > (p[:, None] - self._msa_local))
        keys, vals = [kk], [vv]
        for g in self._msa_groups:
            if g <= 0 or T % g or T // g < 2:
                continue                      # 组数 < 2 时这条尺度没有信息量
            n_g = T // g
            keys.append(kk.reshape(B, H, n_g, g, hd).mean(3))
            vals.append(vv.reshape(B, H, n_g, g, hd).mean(3))
            gi = torch.arange(n_g, device=dev)
            mask = torch.cat([mask, ((gi[None, :] + 1) * g - 1) < p[:, None]], dim=1)
        K = torch.cat(keys, 2)
        V = torch.cat(vals, 2)
        y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        return y

    def _block_mask(self, T, S, device):
        """Z-order 块内双向、块间因果的 mask（True = 可见）。

        块的边界是 4^k（1, 4, 16, 64, 256, ...），对应 Z-order 下
        2^k × 2^k 的子块 —— 这正是 VAR 的"尺度"。
        token i 可见 [0, bound(i))，bound(i) = 最小的 4 的幂 ≥ i+1。
        """
        need = torch.arange(1, T + 1, device=device, dtype=torch.long)   # 1..T
        bound = torch.full((T,), T, dtype=torch.long, device=device)
        b = 1
        while b < T:
            cand = torch.full((T,), b, dtype=torch.long, device=device)
            bound = torch.where(need <= b, torch.minimum(bound, cand), bound)
            b *= 4
        j = torch.arange(S, device=device, dtype=torch.long)
        m = j[None, :] < bound[:, None]        # (T, S) bool，True = 可见
        return m[None, None, :, :]             # (1, 1, T, S)

    def _ms_block_mask(self, T, device):
        """多尺度残差的 block-causal mask：(1,1,T,S)，True = 可见。

        规则与 VAR 的 `lvl(i) >= lvl(j)` 完全一致：
            token i 可 attend token j  ⟺  scale(i) >= scale(j)
        ⇒ 尺度内双向（同尺度是残差，近似独立），跨尺度因果。

        与 _block_mask（Z-order 版）的区别见 cfg.mask_mode 的注释。
        """
        pns = self._patch_nums
        if not pns:
            return None
        # 建位置 → 尺度 的映射（长度 = Σ pn²，应与 T 一致）
        scale_of = []
        for si, pn in enumerate(pns):
            scale_of.extend([si] * (pn * pn))
        total = len(scale_of)
        if total != T:
            # 长度不匹配说明用了别的 token 布局 —— 静默退化成因果，
            # 但必须让调用方知道（否则会以为在用多尺度 mask）
            raise ValueError(
                'mask_mode=msblock 的 patch_nums=%s 总长 %d 与序列长 T=%d 不符；'
                '请确认 token 是按尺度拼接的多尺度序列' % (pns, total, T))
        s = torch.tensor(scale_of, device=device, dtype=torch.long)
        m = s[None, :] <= s[:, None]           # (T, T) bool
        return m[None, None, :, :]             # (1, 1, T, S)

    def _checkerboard_mask(self, T, device):
        """尺度内【棋盘格两相】的严格无泄漏 mask（保底方案，见报告 §10.10）。

        如果 msblock 被 teacher-forcing top-1 判定为泄漏，改用这个：
        每个尺度拆成两相（棋盘格），同相 token 互不相邻 ⇒ 无泄漏。
        AR 步数 5 → 10（仍比 256 步好 25.6×）。
        """
        pns = self._patch_nums
        if not pns:
            return None
        ph, sc = [], []
        for si, pn in enumerate(pns):
            for r in range(pn):
                for c in range(pn):
                    sc.append(si)
                    ph.append((r + c) % 2)
        if len(sc) != T:
            raise ValueError('patch_nums 总长 %d != T %d' % (len(sc), T))
        s = torch.tensor(sc, device=device, dtype=torch.long)
        p = torch.tensor(ph, device=device, dtype=torch.long)
        # 可见条件：(尺度更小) 或 (同尺度且相位相同)
        m = (s[None, :] < s[:, None]) | ((s[None, :] == s[:, None]) & (p[None, :] == p[:, None]))
        return m[None, None, :, :]


class CrossAttention(nn.Module):
    """图像 token 直接 attend 到文本 token 序列（用序列，不是 pooled 向量）。"""

    def __init__(self, cfg):
        super().__init__()
        d, nh = cfg.d_model, cfg.n_head
        self.nh, self.hd = nh, d // nh
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x, ctx):
        B, T, _ = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_norm(self.k(ctx).view(B, L, self.nh, self.hd)).transpose(1, 2)
        v = self.v(ctx).view(B, L, self.nh, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))


# ------------------------------------------------------------------ 主块
class BlockV2(nn.Module):
    def __init__(self, cfg, cond_dim, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        d = cfg.d_model
        cm = cfg.cond_mode
        self.cm = cm
        self.ln1 = RMSNorm(d)
        self.attn = _make_attn(cfg)
        self.ln2 = RMSNorm(d)
        # ---- FFN：SwiGLU 或 L³ ----
        from l3_ffn import L3FFN
        self.is_l3 = bool(getattr(cfg, 'use_l3', False)) and (layer_idx >= getattr(cfg, 'l3_from', 0))
        if self.is_l3:
            self.mlp = L3FFN(d, table_size=cfg.l3_v, k_emb=cfg.l3_k,
                             d_emb=cfg.l3_d_emb, d_up=cfg.l3_d_up,
                             sparse=getattr(cfg, 'l3_sparse', False))
        else:
            self.mlp = SwiGLU(d, cfg.ffn_dim)
        if cm == 'adaln':
            # 每层 6*d 的调制量从 ada_dim 投影出来，ada_dim 由模型级共享瓶颈产出。
            # 直接从 d 投影会多出 6*d*d 参数（10 层 = 15.7M），对 33M 小模型太重。
            self.ada = nn.Linear(cfg.ada_dim, 6 * d)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)
        if cm == 'cross':
            self.ln_c = RMSNorm(d)
            self.cross = CrossAttention(cfg)
        self.ls1 = nn.Parameter(torch.full((d,), 1e-2))
        self.ls2 = nn.Parameter(torch.full((d,), 1e-2))
        if cm == 'cross':
            self.ls3 = nn.Parameter(torch.full((d,), 1e-2))

    def forward(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
        if self.cm == 'adaln' and pooled is not None:
            sa, ba, sm, bm, ga, gm = self.ada(pooled).chunk(6, dim=-1)
            h = self.ln1(x) * (1 + sa.unsqueeze(1)) + ba.unsqueeze(1)
        else:
            h = self.ln1(x)
        x = x + self.ls1 * self.attn(h, cache=cache, append=append)

        # L³ 层吃静态键；SwiGLU 层忽略
        def _ffn(hh):
            return self.mlp(hh, keys) if self.is_l3 else self.mlp(hh)

        if self.cm == 'adaln' and pooled is not None:
            h = self.ln2(x) * (1 + sm.unsqueeze(1)) + bm.unsqueeze(1)
            x = x + self.ls2 * (_ffn(h) * (1 + gm.unsqueeze(1)))
        else:
            x = x + self.ls2 * _ffn(self.ln2(x))

        if self.cm == 'cross' and ctx is not None:
            x = x + self.ls3 * self.cross(self.ln_c(x), ctx)
        return x


def build_gen_caches(n_layer, loop_idx, cap, loop_cfg=None):
    """生成用的 KV cache 分配。

    默认（'shared'）每层一份，与历史行为一致。

    loop_cache='perloop' 时，循环块拿到的是【每轮一份】的 list。原因是训练路径从不传
    cache（forward_logits/forward_masked/forward_hier 都是 None），所以每一轮都是用当轮
    隐状态重算 K/V —— 第 t 轮 attend 的是 loop-t 的 K/V。采样时若共用一份、只有第 0 轮
    写入，第 t>0 轮看到的就是第 0 轮的上下文，训练与推理不一致。
    参考 Nanbeige4.2-3B (arXiv:2607.22083) 与 Looped Latent Attention (arXiv:2607.15456)，
    两者都选择"每轮各自一份"。
    """
    perloop = (loop_cfg is not None
               and getattr(loop_cfg, 'loop_cache', 'shared') == 'perloop')
    n_slots = 1
    if perloop:
        n_slots = max(1, loop_cfg.loop_L_max if loop_cfg.loop_random else loop_cfg.loop_L)
    out = []
    for i in range(n_layer):
        if perloop and i in loop_idx:
            out.append([dict(k=None, v=None, cap=cap, n=0) for _ in range(n_slots)])
        else:
            out.append(dict(k=None, v=None, cap=cap, n=0))
    return out


def run_blocks(blocks, x, loop_idx, loop_times, pooled, ctx, caches, keys=None,
               loop_cfg=None, loop_mods=None):
    """前向所有块；循环块重复跑时**复用缓存**（append=False），否则增量解码会丢上下文。

    keys: (B,T) int64 静态路由键，只有 L³ 层用得上；SwiGLU 层忽略它。

    loop_cfg/loop_mods: 循环 Transformer 的配置与模块（见 loop_transformer.py）。
        传 None 时走【原路径】，行为与之前完全一致（向后兼容）。
    """
    if loop_cfg is not None:
        from loop_transformer import run_blocks_looped
        return run_blocks_looped(blocks, x, loop_idx, loop_cfg, loop_mods,
                                 pooled, ctx, caches, keys)

    for i, blk in enumerate(blocks):
        cache = caches[i] if caches is not None else None
        x = blk(x, pooled=pooled, ctx=ctx, cache=cache, append=True, keys=keys)
        if i in loop_idx:
            for _ in range(max(1, loop_times) - 1):
                x = blk(x, pooled=pooled, ctx=ctx, cache=cache, append=False, keys=keys)
    return x


class FlatHead(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.lin = nn.Linear(d, vocab, bias=False)

    def logits(self, x):
        return self.lin(x), None


class FactorizedHead(nn.Module):
    """vocab = n_cluster * per。返回 (cluster_logits, code_logits)。"""

    def __init__(self, d, vocab, n_cluster):
        super().__init__()
        assert vocab % n_cluster == 0
        self.n_cluster = n_cluster
        self.per = vocab // n_cluster
        self.cl = nn.Linear(d, n_cluster, bias=False)
        self.cd = nn.Linear(d, self.per, bias=False)

    def logits(self, x):
        return self.cl(x), self.cd(x)


# ------------------------------------------------------------------ 配置 / 主模型
@dataclass
class SmallImageConfigV2:
    vocab_size: int = 1024
    d_model: int = 512
    n_layer: int = 10
    n_head: int = 8
    n_kv_head: int = 8
    ffn_dim: int = 2048
    total_tokens: int = 256
    grid: int = 16
    think_tokens: int = 16
    text_dim: int = 512
    ada_dim: int = 128            # adaLN 共享瓶颈维度（控制条件注入的参数量）
    cond_mode: str = 'adaln'
    cond_dropout: float = 0.1
    pos_mode: str = '2d'
    token_order: str = 'raster'
    head_mode: str = 'factorized'
    n_cluster: int = 32
    hier: bool = False            # True = Think 头预测 4x4 粗 token，Gen 头条件于它生成 16x16
    coarse_grid: int = 4
    loop_start: int = 4
    loop_end: int = 7
    loop_times: int = 2
    fuse_qkv: bool = False       # True = q/k/v 三个 Linear 合成一个（实测快 16~26%）
    use_bos: bool = False        # True = 序列前置一个可学习 BOS    #   为什么需要它：原实现训练时用 [x0..x_{T-1}] 预测 [x1..x_T]，
    #   也就是【永远不需要预测第一个 token】。而采样时 generate() 拿
    #   token ID 0 当起点 —— 那是个真实码本条目，模型被迫在一个与目标无关的
    #   前提下生成整张图，之后每步都在分布外。
    #   实测（sample_ar_cond.py）：给 75% 真实前缀能完美延续、50% 良好、
    #   25% 退化、0% 变抽象色块 —— 断点就在冷启动。
    #   开启后：输入 [BOS, x0..x_{T-2}]，目标 [x0..x_{T-1}]，
    #   logits[:, i] 直接预测 tokens[:, i]（序列长度与算力都不变）。

    # ---- L³ 替换 FFN（arXiv:2601.21461）----
    # 本机实测（B=8 T=256 d=512，训练含反向）：
    #   SwiGLU 512→2048→512      260.06 ms   1.00x    3.15M
    #   L³ V=4096 稠密          【230.06 ms   1.13x   21.32M】  ★ 最优
    #   L³ V=64K  稠密           3551.33 ms   0.07x  335.89M   ← 稠密梯度灾难
    #   L³ V=64K  sparse         317.25 ms   0.82x  335.89M
    # ⇒ 默认 V=4096 稠密：比 SwiGLU 快 13%，FFN 容量大 6.8×
    use_l3: bool = False
    # ---- VAR 风格掩码（2026-09-18）----
    # 'causal' = 原光栅序逐 token 因果（默认，向后兼容）
    # 'block'  = Z-order 排列 + 块内双向/块间因果（借 VAR 的核心思想）
    #   动机：实测 T=256 光栅序 causal 下生成 token 相邻同率 42~54%（真实 4.47%）
    #        且逐位置 top-1 越后越差（11.5%→5.7%）⇒ 长程依赖失效
    #   VAR（arXiv 2404.02905, NeurIPS 2024 Best Paper）用"尺度内双向注意力"
    #   把 FID 从 18.65 降到 1.73 ⇒ 本项借其掩码结构。
    #   ⚠️ 需配合 token_order='zorder'（4^k 前缀 = 2^k×2^k 方形子块）
    # 'full'   = 全双向（GRN global refinement 的前提）
    # 'msblock'= 【多尺度残差】的 block-causal：token 按尺度拼接
    #            (patch_nums=(1,2,4,8,16) ⇒ 总长 341)，规则 scale(i) >= scale(j)
    #            ⇒ 尺度内双向、跨尺度因果。
    #   ⚠️ 与 'block' 的关键差别：'block' 是拿 Z-order 前缀当"块"，同块 token 是
    #      图像真实内容、强相关 ⇒ 实测 teacher-forcing top-1 = 88.1%（抄答案）。
    #      'msblock' 的同尺度 token 是【残差】。
    #      实测（probe_ms_leak.py，384 图，条件熵 leak = 1 − H(X|Y)/H(X)）：
    #        单尺度 16x16 : leak 0.369，相邻同码率 5.38%，随机基线 0.155% ⇒ 34.8x
    #        多尺度 尺度16: leak 0.208，相邻同码率 2.53%，随机基线 0.226% ⇒ 11.2x
    #      ⇒ 依赖降低 1.8~3.1 倍，但【不是独立】。最终必须用 teacher-forcing
    #        top-1 判决（88.1% = 泄漏，5.4% = 健康）。
    mask_mode: str = 'causal'
    # 多尺度尺度划分（mask_mode='msblock' 时必须给，如 (1,2,4,8,16)）
    patch_nums: tuple = ()
    # ---- 多尺度压缩注意力（mask_mode='msa'，2026-10-04）----
    # 借 CSA/HCA(arXiv:2606.19348) 沿序列轴压缩 KV 的思路，但组大小按二维网格重定：
    # 光栅序下一行 = grid 个 token，所以 ×grid 是"行摘要"。默认 'grid,4*grid'。
    # ⚠️ 不放 ≥T 的值：全局摘要要整幅图生成完才存在，因果自回归里用不上。
    msa_groups: str = ''
    # 未压缩的局部窗口长度（0 = 默认 2*grid，即两行）
    msa_local: int = 0
    # ---- 邻居融合（2026-09-18，直接针对实测瓶颈）----
    # 实测 16x16 token 流的条件熵（29113 张图）：
    #     H(x|左邻)   = 8.674 bits   ← 单独只解释 1.1 bits
    #     H(x|上邻)   = 8.486 bits   ← 单独只解释 1.3 bits
    #     H(x|左+上)  = 3.384 bits   ← 联合解释 6.4 bits  【强超可加】
    # ⇒ 信息在"两个邻居的联合构型"里，但光栅序下上邻距离 15 步、
    #   zorder 下距离 1~2 步，模型都要靠注意力间接学这个交互。
    #   直接把它们拼进输入 ⇒ 交互变成"一跳"就能算的函数。
    # 因果安全：光栅序下 up=p-grid、left=p-1 必然 < p；Z-order（Morton 码）
    #   对 (r-1,c)/(r,c-1) 单调 ⇒ 邻居序列位置也必然 < p。两者都在因果前缀里，
    #   所以【不可能泄漏】。成本：2×d² = 524K 参数（+1.7%）。
    nbr_fuse: bool = False
    # 邻居窗口半径：0 = 只用上/左两个 tap（向后兼容）；
    #   1 = 因果半平面 4 tap {(0,-1),(-1,-1),(-1,0),(-1,1)}
    #   2 = 更宽（7 tap，含 (-2,*)）
    # 为什么可变：增益该有多大目前【量不准】（见 probe_ctx_gain.py —— 计数式留出 CE
    # 的稀疏偏差随条件变量个数指数增长，加更多上下文反而"更差"，比较无效）。
    # 所以窗口宽度直接靠 A/B 实测决定，而不是靠熵估计。
    nbr_window: int = 0
    l3_v: int = 4096             # 查表大小（做大了稠密梯度会爆）
    l3_k: int = 8                # 每个键分配几个 embedding
    l3_d_emb: int = 128
    l3_d_up: int = 128
    l3_key_mode: str = '4gram'   # 1gram/2gram/4gram/5gram（空间因果邻域）
    l3_from: int = 0             # 从第几层开始用 L³（默认全部）
    l3_sparse: bool = False      # True 则用 sparse=True（需要配 SparseAdam）


class SmallARImageModelV2(nn.Module):
    def __init__(self, cfg: SmallImageConfigV2):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.token_emb = nn.Embedding(cfg.vocab_size, d)
        if cfg.pos_mode == '2d':
            self.row_emb = nn.Parameter(torch.zeros(cfg.grid, d))
            self.col_emb = nn.Parameter(torch.zeros(cfg.grid, d))
            nn.init.normal_(self.row_emb, std=0.02)
            nn.init.normal_(self.col_emb, std=0.02)
        elif cfg.pos_mode == 'ms':
            # 多尺度位置编码（照抄 VAR 的做法）：
            #   pos_emb = 全序列学习式位置嵌入（长度 = Σ pn² = 341）
            #   lvl_emb = 尺度嵌入（长度 = n_scale = 5）
            # 为什么不能用 '2d'：2d 走 self.order[start:start+T]，而 order 只有
            # grid²=256 项，T=341 时长度根本不匹配（会 broadcast 报错）。
            # 也不能只用一个 (row,col) 表：每个尺度的网格边长不同（1/2/4/8/16），
            # 同一 (r,c) 在不同尺度含义不同 ⇒ 必须靠 lvl_emb 区分尺度。
            self.pos_emb = nn.Embedding(cfg.total_tokens, d)
            self.lvl_emb = nn.Embedding(len(cfg.patch_nums), d)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            nn.init.normal_(self.lvl_emb.weight, std=0.02)
            # 位置 → 尺度 的查表（buffer，随 ckpt 走）
            scale_of = []
            for si, pn in enumerate(cfg.patch_nums):
                scale_of.extend([si] * (pn * pn))
            assert len(scale_of) == cfg.total_tokens, \
                'patch_nums 总长 %d != total_tokens %d' % (len(scale_of), cfg.total_tokens)
            self.register_buffer('ms_scale_of',
                                 torch.tensor(scale_of, dtype=torch.long), persistent=False)
        else:
            self.pos_emb = nn.Embedding(cfg.total_tokens, d)

        # ---- 邻居融合：预计算上/左邻居的【序列位置】表 ----
        # 因果安全的证明见 cfg.nbr_fuse 注释；这里额外 assert 一遍，
        # 因为一旦邻居位置 >= 自身位置，训练就会静默地抄答案（最难查的一类 bug）。
        self.nbr_up = self.nbr_left = None
        self.nbr_offsets = []
        self.nbr_linears = None
        # ⚠️ 必须先 register_buffer(None) 再赋值：直接 `self.nbr_up_idx = None`
        # 会在 __dict__ 里占名，之后 register_buffer 会抛
        # "attribute already exists"。注册成 None 之后再赋 Tensor，
        # nn.Module.__setattr__ 会把它写回 _buffers，`.to(device)` 才能跟着走。
        self.register_buffer('nbr_up_idx', None, persistent=False)
        self.register_buffer('nbr_left_idx', None, persistent=False)
        if getattr(cfg, 'nbr_fuse', False):
            if cfg.patch_nums:
                raise ValueError('nbr_fuse 目前只支持单尺度布局（多尺度每个尺度网格不同）')
            g = cfg.grid
            order = make_token_order(g, cfg.token_order)
            pos_of = [0] * (g * g)
            for p, cell in enumerate(order):
                pos_of[cell] = p
            # 因果半平面的偏移集合（dr<0 整行可见；dr==0 只有左侧可见）
            R = int(getattr(cfg, 'nbr_window', 0) or 0)
            if R > 0:
                offs = [(dr, dc) for dr in range(-R, 1) for dc in range(-R, R + 1)
                        if dr < 0 or (dr == 0 and dc < 0)]
            else:
                offs = [(0, -1), (-1, 0)]        # 向后兼容：只有上、左
            # 每个偏移一张序列位置表；越界用 -1
            # ★ 因果过滤：只保留【序列位置严格小于自身】的偏移。
            #   实测教训：zorder 下"右上" (-1,+1) 的 Morton 码比自身【更大】
            #   （r 减 1 但 c 加 1），序列位置在后面 ⇒ 用它就是泄漏。
            #   raster 下 (-1,+1) 对应 p-grid+1 < p，是安全的。
            #   ⇒ 安全的偏移集合【依赖于 token 顺序】，必须逐个筛。
            idx_tables, kept_offs, dropped = [], [], []
            for (dr, dc) in offs:
                tab = []
                for p, cell in enumerate(order):
                    r, c = divmod(cell, g)
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < g and 0 <= cc < g:
                        tab.append(pos_of[rr * g + cc])
                    else:
                        tab.append(-1)
                if any(q >= p for p, q in enumerate(tab) if q >= 0):
                    dropped.append((dr, dc))
                    continue
                idx_tables.append(tab)
                kept_offs.append((dr, dc))
            if dropped:
                print('[nbr_fuse] ⚠️ token_order=%s 下这些偏移【不因果安全】已丢弃: %s'
                      % (cfg.token_order, dropped))
            if not kept_offs:
                raise ValueError('nbr_fuse: 没有任何因果安全的偏移（order=%s）'
                                 % cfg.token_order)
            self.nbr_offsets = kept_offs
            self.register_buffer('nbr_idx_tables',
                                 torch.tensor(idx_tables, dtype=torch.long),
                                 persistent=False)
            # 每个偏移一个线性层（比共享更有表达力；成本 n_tap * d^2）
            self.nbr_linears = nn.ModuleList([nn.Linear(d, d, bias=False)
                                              for _ in kept_offs])
            for lin in self.nbr_linears:
                nn.init.normal_(lin.weight, std=0.02)
            # 边界缺邻居用可学习的"缺失"向量，而不是 0
            # （否则模型分不清"邻居嵌入恰好是 0"和"没有邻居"）
            self.nbr_null = nn.Parameter(torch.zeros(len(kept_offs), d))
            self.nbr_up = self.nbr_linears[0]
            self.nbr_left = self.nbr_linears[-1] if len(kept_offs) > 1 else None

        if cfg.cond_mode in ('add', 'cross'):
            self.cond_proj = nn.Linear(cfg.text_dim, d)
        elif cfg.cond_mode == 'adaln':
            # 共享瓶颈：text_dim -> ada_dim，再分发给每层的 6*d 调制量
            self.cond_proj = nn.Sequential(nn.Linear(cfg.text_dim, cfg.ada_dim), nn.SiLU())
        else:
            self.cond_proj = None
        cond_dim = cfg.ada_dim if cfg.cond_mode == 'adaln' else d
        self.blocks = nn.ModuleList([BlockV2(cfg, cond_dim, layer_idx=i)
                                     for i in range(cfg.n_layer)])
        mk = (lambda: FactorizedHead(d, cfg.vocab_size, cfg.n_cluster)) if cfg.head_mode == 'factorized' \
            else (lambda: FlatHead(d, cfg.vocab_size))
        self.think_head = mk()
        self.gen_head = mk()
        self.loop_idx = list(range(cfg.loop_start, min(cfg.loop_end, cfg.n_layer - 1) + 1))
        self.register_buffer('order', torch.tensor(make_token_order(cfg.grid, cfg.token_order)),
                             persistent=False)
        # ---- 循环 Transformer（可选，默认关闭以保持向后兼容）----
        # 用 set_loop() 挂载。挂载后所有 run_blocks 调用会自动走 looped 路径。
        self.loop_cfg = None
        self.loop_mods = None
        # 专属 BOS：词嵌入 + 位置嵌入各一个向量（不占词表、不污染 token 0）
        self.use_bos = bool(getattr(cfg, 'use_bos', False))
        if self.use_bos:
            self.bos_emb = nn.Parameter(torch.zeros(d))
            self.bos_pos = nn.Parameter(torch.zeros(d))
            nn.init.normal_(self.bos_emb, std=0.02)
            nn.init.normal_(self.bos_pos, std=0.02)
        if cfg.hier:
            # coarse-to-fine：序列变成 [coarse(16) ; fine(256)]，
            # 粗层用自己的 4x4 位置嵌入，并用一个 level 嵌入区分粗细
            g, cg = cfg.grid, cfg.coarse_grid
            self.c_row = nn.Parameter(torch.zeros(cg, d))
            self.c_col = nn.Parameter(torch.zeros(cg, d))
            nn.init.normal_(self.c_row, std=0.02)
            nn.init.normal_(self.c_col, std=0.02)
            self.level_emb = nn.Parameter(torch.zeros(3, d))   # 0=BOS, 1=粗, 2=细
            nn.init.normal_(self.level_emb, std=0.02)
            # 粗 token 从细网格按步长 grid//coarse_grid 采样（光栅序下标）
            step = g // cg
            idx = [(r * step) * g + (c * step) for r in range(cg) for c in range(cg)]
            self.register_buffer('coarse_idx', torch.tensor(idx), persistent=False)

    # ---------------- 循环 Transformer（可选）----------------
    def set_loop(self, loop_cfg):
        """挂载循环 Transformer（见 loop_transformer.py）。

        必须在 optimizer 构造【之前】调用，否则 loop_mods 的参数不在优化器里。
        传 None 表示关闭（回到原始路径，行为与之前完全一致）。
        """
        from loop_transformer import make_loop_modules
        self.loop_cfg = loop_cfg
        if loop_cfg is None:
            self.loop_mods = None
            return
        self.loop_mods = make_loop_modules(self.cfg, loop_cfg)
        ref = next(self.parameters())
        self.loop_mods.to(device=ref.device, dtype=ref.dtype)
        n_new = sum(p.numel() for p in self.loop_mods.parameters())
        print('[loop] 循环 Transformer 已挂载: L=%s L_max=%s random=%s '
              'step_enc=%s cross_res=%s  新增参数 %d'
              % (loop_cfg.loop_L, loop_cfg.loop_L_max, loop_cfg.loop_random,
                 loop_cfg.step_enc, loop_cfg.cross_res, n_new), flush=True)

    # ---------------- 层级模式 ----------------
    def coarse_of(self, fine):
        """(B,256) 细 token -> (B,16) 粗 token（按 coarse_idx 采样）。"""
        return fine[:, self.coarse_idx]

    # ---------------- 位置 ----------------
    def coords(self, start, T, device):
        g = self.cfg.grid
        p = self.order[start:start + T]
        return (p // g).to(device), (p % g).to(device)

    def embed(self, tokens, pos_offset=0, nbr_buf=None):
        B, T = tokens.shape
        x = self.token_emb(tokens)
        if self.cfg.pos_mode == '2d':
            r, c = self.coords(pos_offset, T, tokens.device)
            x = x + self.row_emb[r] + self.col_emb[c]
        elif self.cfg.pos_mode == 'ms':
            pos = torch.arange(pos_offset, pos_offset + T, device=tokens.device)
            x = x + self.pos_emb(pos) + self.lvl_emb(self.ms_scale_of[pos])
        else:
            x = x + self.pos_emb(torch.arange(pos_offset, pos_offset + T, device=tokens.device))

        # ---- 邻居融合：把因果半平面内若干邻居的 token 嵌入直接拼进来 ----
        # 实测依据见 cfg.nbr_fuse 的注释（H(x|左+上) 比 H(x|左)、H(x|上) 低 5 bits
        # ⇒ 信息在两邻居的联合构型里，靠 15 步注意力间接学太慢）。
        if self.nbr_linears is not None:
            dev = tokens.device
            tabs = self.nbr_idx_tables[:, pos_offset:pos_offset + T]      # (n_tap, T)
            for ti in range(tabs.shape[0]):
                ti_idx = tabs[ti].to(dev)
                ok = (ti_idx >= 0)[None, :, None]
                if nbr_buf is not None:
                    e = nbr_buf[:, ti_idx.clamp(min=0)]
                else:
                    e = self.token_emb(tokens[:, ti_idx.clamp(min=0)])
                e = torch.where(ok, e, self.nbr_null[ti].view(1, 1, -1))
                x = x + self.nbr_linears[ti](e)
        return x

    def embed_bos(self, tokens, pos_offset=0):
        """带 BOS 的输入嵌入（use_bos=True 时用）。

        tokens: (B,T) 完整真实序列 [x0..x_{T-1}]
        返回:   (B,T,d) = [BOS, x0, ..., x_{T-2}]
                —— x0..x_{T-2} 位于它们本来的空间位置 order[0..T-2]，
                   BOS 占序列第 0 位、用专属嵌入，不占空间位置。
        于是 logits[:, i] 直接预测 tokens[:, i]。
        """
        B, T = tokens.shape
        # 前 T-1 个真实 token 走正常嵌入（空间位置 order[0..T-2]）
        x = self.embed(tokens[:, :-1], pos_offset=pos_offset)
        bos = (self.bos_emb + self.bos_pos).view(1, 1, -1).expand(B, 1, -1)
        return torch.cat([bos, x], dim=1)

    def prep_cond(self, text_cond, B, device):
        cm = self.cfg.cond_mode
        if cm == 'none' or self.cond_proj is None:
            return None, None
        if text_cond is None:
            text_cond = torch.zeros(B, self.cfg.text_dim, device=device)
        pooled = text_cond.mean(dim=1) if text_cond.dim() == 3 else text_cond
        pooled = self.cond_proj(pooled)
        ctx = self.cond_proj(text_cond) if (cm == 'cross' and text_cond.dim() == 3) else None
        return pooled, ctx

    # ---------------- 训练前向 ----------------
    def forward_logits(self, tokens, text_cond=None):
        """tokens: (B,T) 完整真实 token 序列 [x0..x_{T-1}]

        返回 (think_logits, gen_logits)，两者形状同 (B,T,...)。
        预测语义取决于 use_bos：
          use_bos=False（旧）: logits[:, i] 预测 tokens[:, i+1]   -> 训练要移位
          use_bos=True （新）: logits[:, i] 预测 tokens[:, i]     -> 训练【不移位】
                              （BOS 已提供 shift，再手动移位会错位）
        """
        cfg = self.cfg
        B, T = tokens.shape
        x = self.embed_bos(tokens) if self.use_bos else self.embed(tokens)
        keys = self._l3_keys(tokens) if getattr(cfg, 'use_l3', False) else None
        pooled, ctx = self.prep_cond(text_cond, B, tokens.device)
        if self.training and cfg.cond_dropout > 0 and pooled is not None:
            keep = (torch.rand(B, 1, device=x.device) >= cfg.cond_dropout).float()
            pooled = pooled * keep
            if ctx is not None:
                ctx = ctx * keep.unsqueeze(-1)
        x = run_blocks(self.blocks, x, self.loop_idx, cfg.loop_times, pooled, ctx, None, keys, loop_cfg=self.loop_cfg, loop_mods=self.loop_mods)
        return self.think_head.logits(x[:, :cfg.think_tokens]), self.gen_head.logits(x)

    def _l3_keys(self, tokens):
        """算 L³ 的静态路由键。use_bos 时序列前面多一位，键也跟着右移并补 0。"""
        from l3_ffn import build_keys
        return build_keys(tokens, self.cfg.grid, self.cfg.l3_v,
                          self.cfg.l3_key_mode, use_bos=self.use_bos)

    # ---------------- MaskGIT 离散扩散前向 ----------------
    def forward_masked(self, tokens, text_cond=None):
        """MaskGIT (arXiv 2202.04200) 风格的前向：**不移位、不 BOS**。

        tokens: (B,T) 其中被掩码的位置已经是 MASK id。
        返回 gen logits (B,T,V)，**logits[:, i] 直接预测 tokens[:, i]**。

        为什么不能走 forward_logits：
          - use_bos=True 时 embed_bos 会【前置 BOS 并丢掉最后一个 token】⇒ 位置 i 装的是
            token i-1，那是对因果模型的移位约定；MaskGIT 要的是位置 i 装 token i（可能是
            MASK），从【双向】上下文预测它自己。
          - use_bos=False 时约定是 logits[:,i] 预测 token i+1，也要移位。
        所以这里直接用 embed(tokens)，长度 T、位置对齐。

        ★ 为什么这样【不可能泄漏】（这正是我们踩过的坑）：
          位置 i 的输入是自己的 [MASK] 嵌入 —— 它不含任何关于答案的信息。
          对比 ar_grn256（全双向 + GRN 随机替换、loss 算在所有位置）：
          模型能看到自己那一格的真实 token ⇒ 学会抄自己 ⇒ val 0.97 / top-1 79.6% 全是假的。
          MaskGIT 把【loss 只算在被掩码的位置】+ 输入是 MASK ⇒ 按构造没法抄。
        """
        cfg = self.cfg
        B, T = tokens.shape
        x = self.embed(tokens)
        keys = self._l3_keys(tokens) if getattr(cfg, 'use_l3', False) else None
        pooled, ctx = self.prep_cond(text_cond, B, tokens.device)
        x = run_blocks(self.blocks, x, self.loop_idx, cfg.loop_times, pooled, ctx,
                       None, keys, loop_cfg=self.loop_cfg, loop_mods=self.loop_mods)
        # ⚠️ FlatHead.logits / FactorizedHead.logits 返回的是元组 (logits, ...)，
        # 必须解包 —— 直接索引会报 "tuple indices must be integers"。
        lg = self.gen_head.logits(x)
        return lg[0] if isinstance(lg, (tuple, list)) else lg

    @torch.no_grad()
    def generate_masked(self, total=None, n_iter=16, mask_id=None, temperature=1.0,
                        batch=1, seed=None, device=None, confidence='max'):
        """MaskGIT 迭代解掩码采样。

        从【全 MASK】开始，每次前向预测所有被掩码位置，按置信度保留 top-k，
        其余重新置为 MASK。k 用余弦调度 γ(r)=cos(π/2·r)（MaskGIT 原文）。

        与因果 AR 的关键差别：**只要 n_iter 次前向**（而不是 T 次），
        且每次都能看【整张图】的双向上下文。
        """
        cfg = self.cfg
        was = self.training
        self.eval()
        dev = device or next(self.parameters()).device
        T = int(total or cfg.total_tokens)
        mask_id = int(mask_id if mask_id is not None else cfg.vocab_size - 1)
        V = cfg.vocab_size - 1                     # 真实码只有前 V 个
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.full((batch, T), mask_id, dtype=torch.long, device=dev)
        for t in range(n_iter):
            lg = self.forward_masked(x, None)[..., :V] / max(temperature, 1e-6)
            # ★★ 2026-09-19 修 BUG：必须从预测分布里【采样】，不能取 argmax ★★
            #   症状：batch=16 采出来的 16 张图【逐像素完全相同】。
            #   根因：所有样本的起点都是同一个"全 MASK"状态，而选择
            #   （top-k 按置信度）和取值（argmax）都是确定性的
            #   ⇒ 整个 batch 必然退化成同一个确定性函数值，【多样性恒为 0】。
            #   真正的 MaskGIT 是在每个位置【按预测分布抽样】token，
            #   再用"所选 token 的概率"当置信度决定谁先定稿。
            p = torch.softmax(lg, dim=-1)
            _B = x.shape[0]
            pred = torch.multinomial(p.view(-1, V), 1).view(_B, T)
            if confidence == 'max':
                # MaskGIT 原文的 confidence = 被选中 token 的预测概率
                conf = p.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            else:                                  # 用熵的负值当置信度
                conf = (p * torch.log(p.clamp_min(1e-12))).sum(-1)   # = -H
            # 已经解开的强制保留
            conf = torch.where(x != mask_id, torch.full_like(conf, 1e9), conf)
            # 这一步之后应保留多少个 MASK（余弦调度）
            n_keep = int(math.floor(math.cos(math.pi / 2 * (t + 1) / n_iter) * T))
            n_keep = max(0, min(T, n_keep))
            k = T - n_keep                          # 这一步之后要有 k 个非 MASK
            if k >= T:
                x = pred
            else:
                idx = conf.topk(k, dim=-1).indices
                newx = torch.full_like(x, mask_id)
                newx.scatter_(1, idx, pred.gather(1, idx))
                x = newx
        if was:
            self.train()
        return x

    @torch.no_grad()
    def masked_loss(self, tokens, mask, mask_id, generator=None):
        """MaskGIT 的验证损失：用【同样的掩码方式】评估（loss 只算被掩码的位置）。

        ⚠️ 绝不能用干净输入评估 —— 全双向下会退化成"抄自己"（ar_grn256 的教训）。
        """
        V = self.cfg.vocab_size - 1
        x_in = torch.where(mask, torch.full_like(tokens, int(mask_id)), tokens)
        lg = self.forward_masked(x_in, None)[..., :V]
        if not bool(mask.any()):
            return None
        return F.cross_entropy(lg[mask], tokens[mask])

    # ---------------- 层级前向（coarse-to-fine）----------------
    def embed_hier(self, coarse, fine, bos_id=0):
        """序列布局：[BOS(1) ; coarse(16) ; fine(256)] = 273

        这样对齐后：
          think 头取位置 0..15  (BOS + coarse[0..14]) -> 预测 coarse[0..15]
          gen   头取位置 16..271 (coarse[15] + fine[0..254]) -> 预测 fine[0..255]
        BOS 的存在让"第一个粗 token 由谁来预测"有了着落，否则只能退化成恒等映射。
        """
        cfg = self.cfg
        g, cg = cfg.grid, cfg.coarse_grid
        B = fine.shape[0]
        dev = fine.device
        bos = torch.zeros(B, 1, dtype=torch.long, device=dev) + bos_id
        eb = self.token_emb(bos) + self.level_emb[0]
        ci = self.coarse_idx
        ec = self.token_emb(coarse)
        ef = self.token_emb(fine)
        if cfg.pos_mode == '2d':
            # 粗位置的行列要用它自己的序号 k (0..cg*cg-1)，不能用 coarse_idx
            # （后者是细网格下标 0..255，拿去索引只有 cg 项的 c_row 会越界）
            k = torch.arange(coarse.shape[1], device=dev)
            ec = ec + self.c_row[k // cg] + self.c_col[k % cg]
            fi = self.order
            ef = ef + self.row_emb[fi // g] + self.col_emb[fi % g]
        else:
            ec = ec + self.pos_emb(torch.arange(1, 1 + cg * cg, device=dev))
            ef = ef + self.pos_emb(torch.arange(1 + cg * cg, 1 + cg * cg + g * g, device=dev))
        return torch.cat([eb, ec + self.level_emb[1], ef + self.level_emb[2]], dim=1)

    def forward_hier(self, fine, text_cond=None):
        cfg = self.cfg
        coarse = self.coarse_of(fine)
        B = fine.shape[0]
        x = self.embed_hier(coarse, fine)
        pooled, ctx = self.prep_cond(text_cond, B, fine.device)
        if self.training and cfg.cond_dropout > 0 and pooled is not None:
            keep = (torch.rand(B, 1, device=x.device) >= cfg.cond_dropout).float()
            pooled = pooled * keep
            if ctx is not None:
                ctx = ctx * keep.unsqueeze(-1)
        x = run_blocks(self.blocks, x, self.loop_idx, cfg.loop_times, pooled, ctx, None, loop_cfg=self.loop_cfg, loop_mods=self.loop_mods)
        nc = self.cfg.coarse_grid ** 2
        nf = self.cfg.grid ** 2
        think = self.think_head.logits(x[:, :nc])          # 16 logits -> coarse
        gen = self.gen_head.logits(x[:, nc:nc + nf])       # 256 logits -> fine
        return think, gen

    @torch.no_grad()
    def generate_hier(self, text_cond=None, top_k=100, temperature=1.0, batch=None, seed=None):
        """两阶段采样：先出 4x4 粗图，再条件于它出 16x16 细图。"""
        cfg = self.cfg
        was_training = self.training
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        B = batch or (text_cond.shape[0] if text_cond is not None else 1)
        dev = next(self.parameters()).device
        nc, nf = cfg.coarse_grid ** 2, cfg.grid ** 2
        pooled, ctx = self.prep_cond(text_cond, B, dev)

        def pick(lg):
            lg = lg / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, _ = torch.topk(lg, min(top_k, lg.shape[-1]), dim=-1)
                lg = lg.masked_fill(lg < v[:, [-1]], -1e9)
            return torch.multinomial(F.softmax(lg, dim=-1), 1)

        # ---- 阶段 1：生成 16 个粗 token ----
        # 序列在生成期是 [BOS, coarse(已生成 k 个)]，think 头在最后一个位置给出下一个粗 token
        coarse = torch.zeros(B, 0, dtype=torch.long, device=dev)
        for _ in range(nc):
            xb = self._prefix_embed(coarse)
            xb = run_blocks(self.blocks, xb, self.loop_idx, cfg.loop_times, pooled, ctx, None, loop_cfg=self.loop_cfg, loop_mods=self.loop_mods)
            th = self.think_head.logits(xb)
            lg = th[0] if isinstance(th, tuple) else th
            coarse = torch.cat([coarse, pick(lg[:, -1])], dim=1)

        # ---- 阶段 2：条件于粗图生成 256 个细 token ----
        # 序列 [BOS, coarse(16), fine(已生成 k 个)]，gen 头作用在位置 >= nc 上：
        #   fine 为空时该切片只剩最后一个粗位置，正好负责预测 fine[0]
        fine = torch.zeros(B, 0, dtype=torch.long, device=dev)
        for _ in range(nf):
            xb = self._prefix_embed(coarse, fine)
            xb = run_blocks(self.blocks, xb, self.loop_idx, cfg.loop_times, pooled, ctx, None, loop_cfg=self.loop_cfg, loop_mods=self.loop_mods)
            gh = self.gen_head.logits(xb[:, nc:])
            lg = gh[0] if isinstance(gh, tuple) else gh
            fine = torch.cat([fine, pick(lg[:, -1])], dim=1)
        if was_training:
            self.train()
        return fine

    def _prefix_embed(self, coarse, fine=None):
        """按已有前缀构造嵌入（生成期用，不做 padding）。"""
        cfg = self.cfg
        g, cg = cfg.grid, cfg.coarse_grid
        dev = coarse.device
        B = coarse.shape[0]
        parts = [self.token_emb(torch.zeros(B, 1, dtype=torch.long, device=dev)) + self.level_emb[0]]
        if coarse.shape[1]:
            k = torch.arange(coarse.shape[1], device=dev)
            ec = self.token_emb(coarse)
            if cfg.pos_mode == '2d':
                ec = ec + self.c_row[k // cg] + self.c_col[k % cg]
            parts.append(ec + self.level_emb[1])
        if fine is not None and fine.shape[1]:
            fi = self.order[:fine.shape[1]]
            ef = self.token_emb(fine)
            if cfg.pos_mode == '2d':
                ef = ef + self.row_emb[fi // g] + self.col_emb[fi % g]
            parts.append(ef + self.level_emb[2])
        return torch.cat(parts, dim=1)

    # ---------------- 推理（KV Cache）----------------
    @torch.no_grad()
    def generate(self, text_cond=None, steps=None, top_k=100, temperature=1.0,
                 guidance_scale=1.0, cluster_map=None, seed=None, batch=None):
        cfg = self.cfg
        was_training = self.training
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        # cond 为 None 时必须显式给 batch，否则只会生成 1 张
        B = batch or (text_cond.shape[0] if text_cond is not None else 1)
        total = steps or cfg.total_tokens
        use_cfg = guidance_scale is not None and guidance_scale > 1.0
        dev = next(self.parameters()).device

        def run(cond, tokens):
            """tokens: (B,t) 全序列；返回最后一步的 vocab logits (B,vocab)。"""
            # 预分配 KV cache（cap=total+2 容下 BOS/prefill 的所有位置）。
            # 走预分配路径就不必每步 aten::cat —— 实测 cat 占生成 14.5%。
            caches = build_gen_caches(cfg.n_layer, self.loop_idx, total + 2, self.loop_cfg)
            x = self.embed(tokens, pos_offset=0)
            pooled, ctx = self.prep_cond(cond, B, dev)
            x = run_blocks(self.blocks, x, self.loop_idx, cfg.loop_times, pooled, ctx, caches, loop_cfg=self.loop_cfg, loop_mods=self.loop_mods)
            cl, cd = self.gen_head.logits(x[:, -1:])
            return self.merge(cl, cd, cluster_map)[:, -1], caches

        # ---- L³ 生成期：维护 token 缓冲，逐步算静态键 ----
        use_l3 = bool(getattr(cfg, 'use_l3', False))
        g = cfg.grid
        tokbuf = torch.zeros(B, total, dtype=torch.int64, device=dev) if use_l3 else None

        def l3_key_at(p, nxt=None):
            """算空间位置 p 的静态键。nxt 为 None 时从 tokbuf 取（回填场景）。"""
            if nxt is not None:
                tokbuf[:, p:p + 1] = nxt
            e = tokbuf[:, p]
            d = tokbuf[:, p - 1] if (p % g) != 0 else torch.zeros_like(e)
            up = p - g
            b = tokbuf[:, up] if up >= 0 else torch.zeros_like(e)
            c = tokbuf[:, up + 1] if (up >= 0 and (p % g) != (g - 1)) else torch.zeros_like(e)
            if cfg.l3_key_mode == '1gram':
                h = e
            elif cfg.l3_key_mode == '2gram':
                h = (e * 1000003 + d) % 2147483647
            else:
                h = (e * 1000003 + d * 1000033 + b * 1000037 + c * 1000039) % 2147483647
            return (h % cfg.l3_v).view(B, 1)

        def run_emb(cond, emb, keys=None):
            """emb: (B,t,d) 已嵌入的输入（BOS 用，它没有 token id）。"""
            caches = build_gen_caches(cfg.n_layer, self.loop_idx, total + 2, self.loop_cfg)
            pooled, ctx = self.prep_cond(cond, B, dev)
            x = run_blocks(self.blocks, emb, self.loop_idx, cfg.loop_times, pooled, ctx, caches, keys, loop_cfg=self.loop_cfg, loop_mods=self.loop_mods)
            cl, cd = self.gen_head.logits(x[:, -1:])
            return self.merge(cl, cd, cluster_map)[:, -1], caches

        def run_cached(cond, tok, caches, pos, key):
            x = self.embed(tok, pos_offset=pos, nbr_buf=emb_buf)
            pooled, ctx = self.prep_cond(cond, B, dev)
            x = run_blocks(self.blocks, x, self.loop_idx, cfg.loop_times, pooled, ctx, caches, key, loop_cfg=self.loop_cfg, loop_mods=self.loop_mods)
            cl, cd = self.gen_head.logits(x[:, -1:])
            return self.merge(cl, cd, cluster_map)[:, -1]

        # ---- 邻居融合的生成路径 ----
        # 训练路径从 tokens 里 gather 上/左邻居，但 KV-cache 生成时每次只喂 1 个
        # 新 token ⇒ 必须自己维护一个"已生成 token 的嵌入"缓冲区。
        # 因果性：上/左邻居的序列位置必然 < pos（见 cfg.nbr_fuse 的证明），
        # 所以走到 pos 时它们一定已经在缓冲区里了 —— 逐位置 assert 兜底。
        emb_buf = None
        if self.nbr_linears is not None:
            emb_buf = torch.zeros(B, total + 2, self.cfg.d_model, device=dev)
            filled = torch.zeros(total + 2, dtype=torch.bool, device=dev)

        def nbr_store(pos, tok):
            """把刚生成的 token 的嵌入写进缓冲区，并断言所有邻居都已生成"""
            if emb_buf is None:
                return
            emb_buf[:, pos] = self.token_emb(tok[:, 0].long())
            filled[pos] = True
            for ti in range(self.nbr_idx_tables.shape[0]):
                q = int(self.nbr_idx_tables[ti, pos])
                if q >= 0:
                    assert filled[q], ('位置 %d 的邻居 %d（偏移 %s）还没生成'
                                       % (pos, q, self.nbr_offsets[ti]))

        # 预填。
        # use_bos=True ：从专属 BOS 起步 -> 第一个采样就是 x0，token 的空间位置从 0 开始
        # use_bos=False：旧行为，从 token ID 0 起步（这就是冷启动崩坏的根源）
        if self.use_bos:
            bos = (self.bos_emb + self.bos_pos).view(1, 1, -1).expand(B, 1, -1)
            bos_key = torch.zeros(B, 1, dtype=torch.int64, device=dev) if use_l3 else None
            logits, caches = run_emb(text_cond, bos, bos_key)
            uncond_caches = None
            if use_cfg:
                ul, uncond_caches = run_emb(
                    torch.zeros_like(text_cond) if text_cond is not None else None, bos, bos_key)
                logits = ul + guidance_scale * (logits - ul)
        else:
            start = torch.zeros(B, 1, dtype=torch.long, device=dev)
            logits, caches = run(text_cond, start)
            uncond_caches = None
            if use_cfg:
                ul, uncond_caches = run(torch.zeros_like(text_cond) if text_cond is not None else None, start)
                logits = ul + guidance_scale * (logits - ul)

        out = []
        for t in range(total):
            lg = logits / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, _ = torch.topk(lg, min(top_k, lg.shape[-1]), dim=-1)
                lg = lg.masked_fill(lg < v[:, [-1]], -1e9)
            nxt = torch.multinomial(F.softmax(lg, dim=-1), 1)
            out.append(nxt)
            if t == total - 1:
                break
            # 本次采样出的 token 的空间序号：use_bos 时它就是 x_t（第 t 位），
            # 旧路径下它是 x_{t+1}
            pos = t if self.use_bos else t + 1
            key = l3_key_at(pos, nxt) if use_l3 else None
            # 刚采样出的 token 就落在位置 pos；先入缓冲区，再嵌 pos
            # （pos 的上/左邻居位置 < pos，前面几轮已经写过，assert 兜底）
            nbr_store(pos, out[-1])
            logits = run_cached(text_cond, nxt, caches, pos, key)
            if use_cfg:
                # ★ 修复：原来这里漏传了 key（run_cached 签名是 (cond, tok, caches, pos, key)），
                #   于是任何 --guidance > 1 都会抛
                #   TypeError: run_cached() missing 1 required positional argument: 'key'
                #   ⇒ **CFG 采样从来没成功过**，条件化模型也就无法用 CFG 评估。
                ul = run_cached(torch.zeros_like(text_cond) if text_cond is not None else None,
                                nxt, uncond_caches, pos, key)
                logits = ul + guidance_scale * (logits - ul)
        if was_training:
            self.train()
        return torch.cat(out, dim=1)

    def merge(self, cl, cd, cluster_map):
        """把 (cluster_logits, code_logits) 合并成 (B,T,vocab)。"""
        if cd is None:
            return cl
        lp = F.log_softmax(cl, dim=-1).unsqueeze(-1) + F.log_softmax(cd, dim=-1).unsqueeze(-2)
        B, T, nc, per = lp.shape
        return lp.reshape(B, T, nc * per)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
