# -*- coding: utf-8 -*-
"""mhc.py — Manifold-Constrained Hyper-Connections（mHC）实现

论文: arXiv 2512.24880 (DeepSeek, 2025-12)
  HC 把残差流从 C 扩到 n×C，引入可学习的流间混合，
  但破坏了恒等映射性质 ⇒ Amax Gain 高达 3000 ⇒ 训练崩溃
  mHC 把混合矩阵投影到【Birkhoff 多面体】（双随机矩阵）修复:
    ‖H‖₂ ≤ 1（非扩张）+ 乘法封闭（多层复合稳定）

  实现:
    H̃ = α · mat(x̃' φ) + b         ← 由输入动态生成
    H  = Sinkhorn-Knopp(exp(H̃))     ← 投影（20 次迭代）

【本机实测成本】(bench_mhc.py)
  Sinkhorn 20 迭代: 训练 10.7 ms (占步时 0.39%) / 推理 1.09 ms (1.07%)
  n=4 流混合访存:   2.87 ms (占步时 0.11%)
  ⇒ 总开销约 0.5% 训练 / 1.2% 推理
  ⇒ CPU 上比论文报告的 GPU 6.7% 便宜得多（无 kernel 启动开销）

【对循环 Transformer 的意义】
  循环 = 共享层反复迭代 = 很深的网络 ⇒ 必然遇到恒等映射被破坏
  mHC 正是为此设计，且成本极低 ⇒ 是循环的配套设施
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn(logits, iters=20):
    """把 (N, n, n) 的 logits 投影到双随机矩阵（Birkhoff 多面体）。

    论文用 exp + Sinkhorn-Knopp 交替行/列归一化，20 次收敛。

    ⚠️ 两个实现细节（都是踩过的坑）：
      1) 循环必须以【行归一化】结尾，否则行和不是 1
         （交替做时最后一步决定了哪一维精确）
      2) 不能用原地操作（div_）—— 会破坏 autograd 的 backward
         实测原地版只快 5%，不值得为此丢掉可微性
    """
    H = logits.exp()
    for _ in range(iters):
        H = H / H.sum(-1, keepdim=True).clamp_min(1e-9)   # 行归一化
        H = H / H.sum(-2, keepdim=True).clamp_min(1e-9)   # 列归一化
    # 补一次行归一化，保证行和=列和=1（否则行和会偏 2%）
    H = H / H.sum(-1, keepdim=True).clamp_min(1e-9)
    return H


class MHCResidual(nn.Module):
    """mHC 残差模块：把单流残差扩成 n 条并行流，用双随机矩阵混合。

    形式（参考 HC/mHC 论文）：
        h ∈ R^{B,T,n,C}                      n 条并行残差流
        H_res = Sinkhorn(exp(b + α · φ(x̃)))  (B,T,n,n) 动态生成
        h      = einsum('btnc,btmn->btmc', h, H_res)  流间混合
        h[:, :, 0] += f(h[:, :, 0])          计算只走第一条流

    ⚠️ 关于"计算只走第一条流"：
       论文报告额外开销只有 6.7%，且强调是 "memory access overhead" 而非
       compute overhead ⇒ 推断计算不在所有流上重复。
       本实现按此假设，把 n 条流作为【线性旁路】。

    ⚠️ 关于初始化（踩过的坑）：
       α 是【动态部分】的缩放，不能乘到静态偏置上，否则 exp 后趋近均匀分布，
       H 退化成"n 条流的平均" ⇒ 信号被缩小 n 倍、恒等映射被破坏。
       正确做法：静态偏置直接进 logits，α 只缩放动态项。
    """

    def __init__(self, n_streams=4, d_model=512, sinkhorn_iters=20,
                 alpha_init=0.1, eye_logit=6.0):
        super().__init__()
        self.n = n_streams
        self.d = d_model
        self.iters = sinkhorn_iters
        # 动态项: x̃ ∈ R^{n*d} -> n*n 个 logits（初始为 0，靠 eye_logit 起步）
        self.phi = nn.Linear(n_streams * d_model, n_streams * n_streams, bias=False)
        nn.init.zeros_(self.phi.weight)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        # 静态偏置：让 H_res 初始接近单位阵（保持恒等映射）
        eye = torch.eye(n_streams).reshape(-1) * eye_logit
        self.register_buffer('eye_logit_vec', eye)
        self.norm = nn.RMSNorm(n_streams * d_model)

    def init(self, x):
        """(B,T,d) -> (B,T,n,d)，第一条流放原表示，其余置零。"""
        B, T, d = x.shape
        h = x.new_zeros(B, T, self.n, d)
        h[:, :, 0] = x
        return h

    def _mix_matrix(self, h):
        """由当前流状态动态生成 (B,T,n,n) 双随机混合矩阵。

        注意：静态偏置【不乘 alpha】，alpha 只缩放动态项。
        """
        B, T, n, d = h.shape
        xf = self.norm(h.reshape(B, T, n * d))
        dyn = self.phi(xf) * self.alpha
        # ⚠️ 必须【先 reshape 成 (B,T,n,n) 再做 Sinkhorn】——
        #    否则 sinkhorn 会对 (T, n*n) 这两维归一化，完全错误。
        logits = (dyn + self.eye_logit_vec).view(B, T, n, n)
        return sinkhorn(logits, self.iters)

    def mix(self, h, layer_out):
        """流间混合，并把本层计算输出加回第一条流。

        h:         (B,T,n,d) 当前流状态
        layer_out: (B,T,d)   本层 f(h[:,:,0]) 的输出
        """
        B, T, n, d = h.shape
        H = self._mix_matrix(h)                     # (B,T,n,n)
        # bmm 需要 3D，先把 (B,T,n,n) 压成 (B*T,n,n)
        h2 = torch.bmm(H.reshape(B * T, n, n),
                       h.reshape(B * T, n, d)).reshape(B, T, n, d)
        nxt = h2.clone()
        nxt[:, :, 0] = nxt[:, :, 0] + layer_out
        return nxt

    def read(self, h):
        """(B,T,n,d) -> (B,T,d)，取第一条流作为主表示。"""
        return h[:, :, 0]


def mhc_cost_estimate(n_streams=4, d_model=512, batch=8, seq=256,
                      train_step_ms=2706.5, sinkhorn_iters=20):
    """给出成本估算（基于 bench_mhc.py 的实测数据）。"""
    N = batch * seq
    # 实测: N=2048, iters=20 -> 10.67 ms；线性外推到 N=96
    sink_ms = 10671.4 / 2048 * N / 1000.0
    # 实测 n=4 流混合: 2.87 ms (batch=8, seq=256)
    mix_ms = 2868.1 / 1000.0 * (batch * seq) / (8 * 256)
    return sink_ms, mix_ms, 100 * (sink_ms + mix_ms) / train_step_ms
