# -*- coding: utf-8 -*-
"""ms_vq.py — VAR 风格的多尺度残差量化（arXiv 2404.02905）

## 为什么需要它

2026-09-18 的实测教训：我直接把 VAR 的 block-causal mask 抄到单尺度 AR 上，
结果 **teacher-forcing top-1 = 88.1%**（正常应 ~5%），采样只用 26/1024 个码
—— 典型的信息泄漏。根因是：

    VAR 的「同尺度内双向注意力」**依赖多尺度残差量化这个前提**。
    同一尺度的 token 是【残差】，近似互相独立 => 双向不泄漏。
    而我们单尺度下同块的 token 是图像真实内容、强相关 => 双向 = 抄答案。

=> **要真正抄 VAR，必须先抄它的多尺度残差量化。**

## 机制

对编码器输出 f ∈ (B,C,H,W)，逐尺度量化残差：

    f_hat = 0
    for k, pn in enumerate(patch_nums):          # pn = 1, 2, 4, 8, 16
        f_k   = down(f,      pn)                  # (B,C,pn,pn)
        fh_k  = down(f_hat,  pn)
        r_k   = f_k − fh_k                        # 【残差】
        q_k, idx_k = quantize(r_k / rms_k)        # 归一化后匹配（见下）
        f_hat = f_hat + up(q_k * rms_k, H)        # 上采样后累加

总 token 数 = Σ pn² = 1+4+16+64+256 = 341（H=W=16 时）
粗尺度捕捉全局色调/构图，细尺度逐级补细节。

## 与单尺度的关键差别

| | 单尺度 | 多尺度残差 |
|---|---|---|
| token 数 | 256 | 341 |
| 同尺度关系 | 强相关 | **残差，近似独立** |
| 可否用 block-causal | ❌ 泄漏 | **✅ 不泄漏** |
| 自回归步数 | 256 | **5（每步一个尺度）** |
| 信息量 | 256×10 bit | 341×10 bit（+33%） |

## 版本历史（别重犯）

**v1 失败**：自检/训练里每个尺度只用 1/1024 个码，逐尺度 PSNR 完全没有递进
（10.25 10.02 10.31 10.17 10.33，120 步后仍无分层）。两个 bug：

1. **缺 STE**：`q = codebook(argmin(...))` 的 argmin 不可导，梯度根本回不到
   encoder；只有 commit loss 在推 encoder，而 commit 又把 encoder 输出拉向
   f_hat => 正反馈 => 全塌到一个码。
   对照现有 `VQ.forward` 的正确写法：`quant_st = z + (quant − z).detach()`。
2. **缺 EMA**：码本 `requires_grad_(False)`（跟现有 VQ 一致），但现有 VQ 靠
   `self.training` 分支做 EMA + 死码重启来更新码本，我这里【完全没写】。
   => 码本冻结在随机初始化，一步都没动过。

**v2（本文件）**：① 补 STE；② 补 EMA + 死码重启（逐字段对照 `VQ.forward`）；
③ 加 **逐尺度 RMS 归一化** —— 粗尺度残差量级远大于细尺度（差一个数量级），
共享码本时细尺度样本数占 99%（256/341）会完全主导 EMA，粗尺度的重建幅度
必然错。把每个尺度的残差先除以该尺度的运行 RMS 再匹配，码本就变成
**尺度无关**的，共享才真正成立；重建时再乘回去。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleResidualVQ(nn.Module):
    """多尺度残差量化。复用同一个码本，逐尺度量化【归一化后的】残差。

    参数
    ----
    codebook : nn.Embedding(n_codes, dim)   —— 直接复用现有 VQ 的码本
    patch_nums : tuple  每尺度的边长，如 (1, 2, 4, 8, 16)
    beta : float   承诺损失权重（与 VQ 一致 = 0.25）
    decay : float  EMA 衰减（与 VQ 一致 = 0.99）
    per_scale_norm : bool  是否做逐尺度 RMS 归一化（强烈建议 True）
    """

    def __init__(self, codebook, patch_nums=(1, 2, 4, 8, 16),
                 beta=0.25, decay=0.99, epsilon=1e-5, per_scale_norm=True,
                 dead_thresh=0.5, dead_thresh_rel=False, max_restart_frac=0.0):
        super().__init__()
        self.codebook = codebook
        self.patch_nums = tuple(patch_nums)
        self.n_scale = len(self.patch_nums)
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon
        self.per_scale_norm = per_scale_norm
        # 死码判定（见 _ema_step 里的详细教训）
        self.dead_thresh = dead_thresh
        self.dead_thresh_rel = dead_thresh_rel
        self.max_restart_frac = max_restart_frac
        self.last_dead = 0

        # 每个尺度的累积长度（用于切分序列）
        self.begin_ends = []
        cur = 0
        for pn in self.patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn
        self.total = cur

        # EMA 码本（对照 small_vqvae.VQ，但有一个关键差异见下）
        n_codes = codebook.num_embeddings
        self.register_buffer("ema_count", torch.zeros(n_codes))
        # ★ 必须从 0 起步，不能 clone 码本初始化值！
        # 2026-09-18 教训（第四个假设才是对的，前三个都错了）：
        #   EMA 的分母是 `ema_count + ε`。对【从未被选中】的码，ema_count=0
        #   ⇒ denom = ε = 1e-5，而分子 ema_weight = 0.99·W_init
        #   ⇒ 行 = 0.99·W_init / 1e-5 = 【99000 × W_init】—— 初始化值被放大 99000 倍。
        #   单尺度 VQ 里 W_init = ±1/1024（极小），且【无条件】重启所有死码，
        #   所以这个潜伏 bug 被掩盖了（爆掉的行当场就被 zf[rand] 覆盖）。
        #   我做了两件事让它显形：① 把码本行归一化到 L2 范数 1（放大后是 99000
        #   而不是 445）；② 加了 max_restart_frac=2% 限流，导致 400 个死码里
        #   只有 20 个被复活，剩下 380 个带着爆行活下来被用到。
        #   实测：一步之内码本行范数从 1.0 炸到 15.97 / 61899 / 83928，
        #   然后 rms 逐尺度 0.9 → 64 → 4.3e5 → 3.4e9 发散。
        #   ⇒ ema_weight 从 0 起步 + Laplace 平滑分母，两个都必须改。
        self.register_buffer("ema_weight", torch.zeros(n_codes, codebook.embedding_dim))
        # 逐尺度运行 RMS（归一化用；初值 1.0，训练中自动收敛到真实量级）
        self.register_buffer("rms", torch.ones(self.n_scale))

        # per_scale_norm 模式下码本工作在【归一化残差空间】，所以码本行的模长
        # 应该在 1 附近。不归一化的话初始化时重建幅度会错一个数量级
        # （自检 [3] 的 RMSE ~1.0 就是这个），虽然 EMA 几十步后会自己纠回来，
        # 但前几步的巨大 loss 会让 encoder 抖一下。直接对齐到 EMA 的目标空间。
        if per_scale_norm:
            # 目标空间是「逐元素 RMS = 1」的归一化残差（因为 r_k 除以了它自己的
            # RMS）。所以码本行也初始化成【逐元素 RMS = 1】⇒ 行 L2 范数 = √C = 8。
            # 注意：不是 L2 范数 1！范数 1 意味着逐元素 RMS = 1/√C = 0.125，
            # 会让 q*sk 欠 8 倍（实测 step 0 的 qk/sk 正好 = 0.124 = 1/8）。
            with torch.no_grad():
                w = codebook.weight.data
                if w.numel() > 0:
                    w.copy_(F.normalize(w, dim=1) * (codebook.embedding_dim ** 0.5))
                # ema_weight 保持 0（见上面的教训）

    # ---------------- 量化核心 ----------------
    def _match(self, r_norm):
        """归一化残差 → 码本索引。r_norm: (B,C,h,w) → idx (B*h*w,)

        与现有 VQ 完全一致的 L2 归一化余弦匹配。
        """
        B, C, h, w = r_norm.shape
        rf = r_norm.permute(0, 2, 3, 1).reshape(-1, C)
        rn = F.normalize(rf, dim=1)
        wn = F.normalize(self.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (rn @ wn.t())
        return d.argmin(1)

    def _gather(self, idx, shape):
        """idx (N,) → (B,C,h,w)。按码本原始权重取（不乘 RMS）。"""
        B, C, h, w = shape
        return self.codebook(idx).view(B, h, w, C).permute(0, 3, 1, 2)

    def _ema_step(self, rfs, idxs):
        """一次 EMA + 死码重启。rfs/idxs 是【所有尺度拼接后】的扁平张量。

        逐字段对照 small_vqvae.VQ.forward 的 EMA 分支。
        """
        rf = torch.cat(rfs, 0)          # (N_total, C) —— 已归一化的残差
        idx = torch.cat(idxs, 0)        # (N_total,)
        K = self.codebook.num_embeddings
        one_hot = F.one_hot(idx, K).float()
        count = one_hot.sum(0)
        weight = one_hot.t() @ rf
        self.ema_count.mul_(self.decay).add_(count, alpha=1.0 - self.decay)
        self.ema_weight.mul_(self.decay).add_(weight, alpha=1.0 - self.decay)

        # ★ Laplace 平滑分母（VQ-VAE 原论文 / vector-quantize-pytorch 的做法）。
        # 直接用 `ema_count + ε` 会在 ema_count=0 时把分子放大 1/ε = 1e5 倍
        # （这是本文件第 N 次踩的坑，详见 __init__ 里 ema_weight 的注释）。
        # 平滑后：count=0 的码拿到 cluster_size≈ε ⇒ 行 = 0/ε = 0（不是爆行），
        # 随后被死码重启接管；count=n 的码拿到 cluster_size≈n，与朴素写法一致。
        n_total = self.ema_count.sum()
        cluster_size = (self.ema_count + self.epsilon) / \
            (n_total + K * self.epsilon) * n_total
        self.codebook.weight.data.copy_(
            self.ema_weight / cluster_size.clamp(min=self.epsilon).unsqueeze(1))

        # ---- 死码重启 ----
        # 阈值语义：ema_count 收敛到「每步被选中的次数」，所以 0.5 = 平均每两步
        # 选不到一次 ⇒ 死码。这与单尺度 VQ 完全一致。
        #
        # ⚠️⚠️ 2026-09-18 最隐蔽的一个 bug：这里必须用【绝对】阈值，而且必须 < 1.0。
        #   重启时把 `ema_count[dead] = 1.0` 作为种子。所以：
        #     阈值 < 1.0  ⇒ 重启后的码下一步"活着"，能保持 ~35 步（0.99^k 衰减）
        #     阈值 ≥ 1.0  ⇒ 重启后的码（0.99）下一步【立刻又被判死】⇒ 每步全码本翻新
        #   我一度把阈值改成"相对"（0.5 × 每码平均样本数）当改进，结果：
        #     batch  4: thr = 0.5×1.33 = 0.666 < 1.0  ⇒ 重启粘住，dead≈0（看起来很正常！）
        #     batch 16: thr = 0.5×5.33 = 2.66 ≥ 1.0  ⇒ 重启粘不住，dead≈990/1024
        #                                              【每一步翻新 97% 的码本】
        #   ⇒ 症状只在 batch 16 出现，batch 4 完全看不出来。
        #   这就是单尺度 VQ 用绝对 0.5 的真正原因。
        if self.dead_thresh_rel:
            thr = self.dead_thresh * (float(idx.numel()) / max(K, 1))
            if thr >= 1.0 and not getattr(self, '_warned_thr', False):
                self._warned_thr = True
                print('[ms_vq] ⚠️ 相对死码阈值 thr=%.3f >= 1.0：重启种子恒为 1.0，'
                      '重启将粘不住 ⇒ 每步翻新整个码本。建议用绝对阈值 0.5' % thr)
        else:
            thr = self.dead_thresh
        dead = self.ema_count < thr
        n_dead = int(dead.sum().item())
        cap = int(self.max_restart_frac * K)
        if cap > 0 and n_dead > cap:
            cand = torch.nonzero(dead, as_tuple=False).squeeze(1)
            order = torch.argsort(self.ema_count[cand])
            keep = torch.zeros_like(dead)
            keep[cand[order[:cap]]] = True
            dead = keep
            n_dead = cap
        self.last_dead = n_dead
        if n_dead > 0:
            rand = torch.randint(0, rf.shape[0], (n_dead,), device=rf.device)
            self.codebook.weight.data[dead] = rf[rand].clone()
            self.ema_weight[dead] = rf[rand].clone()
            self.ema_count[dead] = 1.0
        return n_dead

    # ---------------- 编码：图 → 多尺度 token ----------------
    def _scale_of(self, k):
        """eval 用的第 k 个尺度归一化因子（跑训练时 EMA 的落点）"""
        if not self.per_scale_norm:
            return 1.0
        return self.rms[k].clamp(min=1e-4)

    def _rms_now(self, r_k):
        """当前 batch 的残差 RMS（detach）。

        ⚠️ 2026-09-18 教训（第三次栽在同一个地方）：一开始我是拿 EMA 的
        `self.rms`（初值 1.0）当归一化因子。但编码器输出的量级不是 1，于是
        `q * rms` 注入 f_hat 的幅度从一开始就错；误差上溯到下一尺度让残差变
        大、rms 跟着变大、注入幅度再变大 —— **正反馈发散**。实测 21 步后
        `rms = [0.92, 1.77, 34.5, 57.2, 5601.9]`，尺度 4 的 RMS 是 5601。
        EMA(decay=0.99) 时间常数 100 步，根本追不上这个发散速度。

        正确做法：用【当前 batch 的 RMS】做归一化（detach），这样
        `r_k / sk` 的 RMS 按构造恒为 1，**不可能发散**。EMA 只作为
        eval/推理时的落点（那时没有 batch 可统计）。
        """
        if not self.per_scale_norm:
            return 1.0
        return r_k.detach().pow(2).mean().sqrt().clamp(min=1e-4)

    def _update_rms(self, k, cur):
        """把当前 batch 的 RMS 记进 EMA（只给推理用）"""
        if not self.per_scale_norm:
            return
        self.rms[k].mul_(self.decay).add_(cur.detach(), alpha=1.0 - self.decay)

    # ---------------- 编码：图 → 多尺度 token ----------------
    def f_to_idx(self, f, straight_through=True, update_ema=None):
        """f: (B,C,H,W) → (idx_list, f_hat, vq_loss)

        idx_list: List[(B, pn²)]
        f_hat:    (B,C,H,W)  —— 从所有尺度累积的重建（带 STE 梯度）
        vq_loss:  标量 —— 承诺损失 × beta / n_scale

        update_ema: None（默认 = self.training and straight_through）
        """
        B, C, H, W = f.shape
        if update_ema is None:
            update_ema = self.training and straight_through

        # ---- pass 1：算残差链 + 匹配索引（前向值，与 pass 2 完全一致）----
        # sks 必须存下来给 pass 2 复用，否则两遍的归一化因子不同、前向值就不一致
        f_hat = torch.zeros_like(f)
        rs, idxs, sks = [], [], []
        for k, pn in enumerate(self.patch_nums):
            f_k = F.interpolate(f, size=(pn, pn), mode='bilinear', align_corners=False)
            fh_k = F.interpolate(f_hat, size=(pn, pn), mode='bilinear', align_corners=False)
            r_k = f_k - fh_k
            if self.training or not self.per_scale_norm:
                sk = self._rms_now(r_k)
                if self.training:
                    self._update_rms(k, sk)
            else:
                sk = self._scale_of(k)          # eval / 推理：用 EMA
            idx_k = self._match(r_k / sk)
            idxs.append(idx_k)
            rs.append(r_k)
            sks.append(sk)
            q_k = self._gather(idx_k, r_k.shape) * sk
            f_hat = f_hat + F.interpolate(q_k, size=(H, W), mode='bilinear',
                                          align_corners=False)

        # ---- EMA（可选）：码本在 pass 1 之后可能已变 ----
        if update_ema:
            with torch.no_grad():
                rfs = [(r / sks[k]).permute(0, 2, 3, 1).reshape(-1, C)
                       for k, r in enumerate(rs)]
                self._ema_step(rfs, idxs)

        # ---- pass 2：用（可能已更新的）码本重建 + STE + 承诺损失 ----
        f_hat = torch.zeros_like(f)
        commit = f.new_zeros(())
        for k, pn in enumerate(self.patch_nums):
            r_k = rs[k]
            sk = sks[k]
            rf = (r_k / sk).permute(0, 2, 3, 1).reshape(-1, C)
            q_raw = self.codebook(idxs[k])                       # (N, C)
            commit = commit + F.mse_loss(rf, q_raw.detach())
            if straight_through:
                q_flat = rf + (q_raw - rf).detach()              # ★ STE
            else:
                q_flat = q_raw
            (Bk, Ck, hk, wk) = r_k.shape
            q_k = q_flat.view(Bk, hk, wk, Ck).permute(0, 3, 1, 2)
            f_hat = f_hat + F.interpolate(q_k * sk, size=(H, W), mode='bilinear',
                                          align_corners=False)

        idxs = [t.view(B, pn * pn) for t, pn in zip(idxs, self.patch_nums)]
        return idxs, f_hat, self.beta * commit / self.n_scale

    # ---------------- 解码：多尺度 token → f_hat ----------------
    @torch.no_grad()
    def idx_to_fhat(self, idxs, H, W, last_k=None):
        """从多尺度 token 重建 f_hat（推理/评估用，无梯度）

        idxs : List[(B, pn²)]  每个尺度按顺序
        last_k: 只用前 k 个尺度（用于"尺度逐步生成"的中间态可视化）
        """
        B = idxs[0].shape[0]
        C = self.codebook.embedding_dim
        f_hat = torch.zeros(B, C, H, W, device=idxs[0].device,
                            dtype=self.codebook.weight.dtype)
        n = self.n_scale if last_k is None else min(last_k, self.n_scale)
        for i in range(n):
            pn = self.patch_nums[i]
            q = self.codebook(idxs[i]).view(B, pn, pn, C).permute(0, 3, 1, 2)
            f_hat = f_hat + F.interpolate(q * self._scale_of(i), size=(H, W),
                                          mode='bilinear', align_corners=False)
        return f_hat

    # ---------------- 诊断 ----------------
    @torch.no_grad()
    def usage(self, idxs):
        """每个尺度的码本利用率。

        ⚠️ 2026-09-18 教训（第 8 个度量陷阱）：不能拿 `unique/1024` 当利用率。
        尺度 0 每张图只有 1 个 token，batch=4 时【最多只能出现 4 个不同的码】，
        除以 1024 得到 0.4% 会被误判成"坍塌"，而真实利用率是 100%。

        正确做法：除以 `min(n_codes, 该尺度本 batch 的可用 token 数)`。

        返回 List[(n_uniq, capacity, util, top_share, verdict)]
        """
        from collections import Counter
        out = []
        for t in idxs:
            flat = t.reshape(-1).tolist()
            n_tok = len(flat)
            n_uniq = len(set(flat))
            capacity = min(self.codebook.num_embeddings, n_tok)
            util = n_uniq / max(capacity, 1)
            top = Counter(flat).most_common(1)[0][1] / max(n_tok, 1)
            # 判据：利用率 < 50% 才算可疑；样本太少（capacity<16）不下结论
            if capacity < 16:
                verdict = '样本太少(仅%d)' % capacity
            elif util < 0.5:
                verdict = '⚠️ 坍塌'
            else:
                verdict = 'OK'
            out.append((n_uniq, capacity, util, top, verdict))
        return out

    def usage_report(self, idxs, patch_nums=None):
        """用法向量化的一行摘要（写日志用）"""
        pn = patch_nums or self.patch_nums
        parts = []
        for i, (n_uniq, cap, util, top, _) in enumerate(self.usage(idxs)):
            parts.append('%d: %d/%d=%.0f%%' % (pn[i], n_uniq, cap, 100 * util))
        return ' '.join(parts) + '  dead=%d' % self.last_dead

    # ---------------- 序列打包/解包（供 AR 使用）----------------
    def pack(self, idxs):
        """把 List[(B,pn²)] 拼成一条长序列 (B, total)"""
        return torch.cat(idxs, dim=1)

    def unpack(self, seq):
        """(B, total) → List[(B,pn²)]"""
        return [seq[:, a:b] for a, b in self.begin_ends]

    def scale_of_position(self, device=None):
        """返回每个序列位置属于哪个尺度 (total,) —— 用于构造 block mask"""
        v = torch.empty(self.total, dtype=torch.long, device=device)
        for i, (a, b) in enumerate(self.begin_ends):
            v[a:b] = i
        return v

    def block_mask(self, device=None):
        """VAR 的 block-causal mask：(total, total) bool，True = 可见

        规则等价于 VAR 的 `lvl(i) >= lvl(j)`：
          token i 可 attend token j  ⟺  scale(i) >= scale(j)
        => 【同尺度内双向】（因为同尺度是残差，独立），跨尺度因果。
        """
        s = self.scale_of_position(device)
        return s[None, :] <= s[:, None]

    def extra_repr(self):
        return 'patch_nums=%s total=%d per_scale_norm=%s' % (
            self.patch_nums, self.total, self.per_scale_norm)


# ---------------------------------------------------------------------------
# 训练用封装：把现有 VQVAE 的 VQ 换成多尺度残差量化
# ---------------------------------------------------------------------------
def default_patch_nums(grid):
    """按编码器输出网格 grid 生成尺度序列。

    ⚠️ 不能超过 grid —— 尺度比编码器网格还细是退化操作（插值放大再量化）。
        grid=16 -> (1,2,4,8,16)   共 341 token
        grid= 8 -> (1,2,4,8)      共  85 token
        grid=12 -> (1,2,3,4,6,12) hmm 12 不是 2 的幂，用 (1,2,4,8) 再补 grid
    统一取「2 的幂，直到 >= grid，截断到 grid」，网格不是 2 的幂时补上 grid 本身。
    """
    pns = []
    p = 1
    while p < grid:
        pns.append(p)
        p *= 2
    pns.append(grid)
    return tuple(pns)


class MSVQVAE(nn.Module):
    """编码器 / 解码器与 VQVAE 完全一致，只把单尺度 VQ 换成多尺度残差量化。

    与 VQVAE 的接口对齐：forward 返回 (xr, idx, vq_loss)，其中 idx 是
    (B, total) 的打包序列，可以直接喂给 AR 做 teacher forcing。
    """

    def __init__(self, embed_dim=64, n_codes=1024, ch=(64, 128, 256, 256),
                 use_attn=False, two_3x3=False, patch_nums=None,
                 per_scale_norm=True, max_restart_frac=0.0, dead_thresh=0.5,
                 dead_thresh_rel=False, size=256):
        super().__init__()
        from small_vqvae import Encoder, Decoder          # 同目录，无循环依赖
        self.ch = tuple(ch)
        self.size = size
        self.embed_dim = embed_dim
        self.n_codes = n_codes
        self.use_attn = use_attn
        self.two_3x3 = two_3x3
        self.enc = Encoder(ch=self.ch, embed_dim=embed_dim,
                           use_attn=use_attn, two_3x3=two_3x3)
        self.dec = Decoder(ch=tuple(reversed(self.ch)), embed_dim=embed_dim,
                           use_attn=use_attn, two_3x3=two_3x3)
        self.vq_codebook = nn.Embedding(n_codes, embed_dim)
        self.vq_codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.vq_codebook.weight.requires_grad_(False)
        grid = self.grid_size
        if patch_nums is None:
            patch_nums = default_patch_nums(grid)
        assert max(patch_nums) <= grid, \
            'patch_nums %s 超出编码器网格 %d' % (patch_nums, grid)
        self.ms = MultiScaleResidualVQ(
            self.vq_codebook, patch_nums=patch_nums, per_scale_norm=per_scale_norm,
            max_restart_frac=max_restart_frac, dead_thresh=dead_thresh,
            dead_thresh_rel=dead_thresh_rel)

    # ---- 与 VQVAE 保持一致的只读属性 ----
    @property
    def down(self):
        return 2 ** len(self.ch)

    @property
    def grid_size(self):
        return self.size // self.down

    @property
    def patch_nums(self):
        return self.ms.patch_nums

    @property
    def total_tokens(self):
        return self.ms.total

    def forward(self, x):
        f = self.enc(x)
        idxs, f_hat, vq_loss = self.ms.f_to_idx(f)
        xr = self.dec(f_hat)
        return xr, self.ms.pack(idxs), vq_loss

    @torch.no_grad()
    def quantize_only(self, f):
        """给 --perc_stopgrad 用：感知损失走这条路（只有解码器收梯度）。

        与 VQ.quantize_only 的语义一致：返回【不带梯度通路的】量化潜变量。
        """
        _, f_hat, _ = self.ms.f_to_idx(f, straight_through=False, update_ema=False)
        return f_hat

    @torch.no_grad()
    def encode_idx(self, x):
        """x → (B, total) 打包 token 序列（离线 tokenize 数据集用）"""
        idxs, _, _ = self.ms.f_to_idx(self.enc(x), straight_through=False,
                                      update_ema=False)
        return self.ms.pack(idxs)


# ---------------------------------------------------------------------------
# 从 checkpoint 重建 MSVQVAE（对齐 small_vqvae.vqvae_from_ckpt 的作用）
# ---------------------------------------------------------------------------
def msvqvae_from_ckpt(ckpt, map_location='cpu'):
    """从 checkpoint 读【全部架构字段】重建 MSVQVAE 并载入权重。

    为什么需要它：项目里已有 7 处硬编码 `VQVAE(embed_dim=..., ch=...)` 踩过
    "state_dict 对不上" 的坑。多尺度又多了 patch_nums / per_scale_norm 两个字段，
    再靠调用方手写参数迟早错。统一走这里。

    返回 (model, ckpt_dict)。ckpt_dict 里的 arch 字段供调用方检查。
    """
    ck = torch.load(ckpt, map_location=map_location, weights_only=False) \
        if isinstance(ckpt, str) else ckpt
    ch = tuple(ck.get('ch') or (64, 128, 256, 256))
    size = int(ck.get('size', 256))
    grid = int(ck.get('grid') or (size // (2 ** len(ch))))
    pns = ck.get('patch_nums')
    if pns is None:
        pns = default_patch_nums(grid)
    m = MSVQVAE(embed_dim=int(ck.get('embed_dim', 64)),
                n_codes=int(ck.get('n_codes', 1024)),
                ch=ch,
                use_attn=bool(ck.get('use_attn', False)),
                two_3x3=bool(ck.get('two_3x3', False)),
                patch_nums=tuple(pns),
                per_scale_norm=bool(ck.get('per_scale_norm', True)),
                size=size)
    sd = ck.get('model', ck)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print('[msvqvae_from_ckpt] 缺 key %d 个: %s' % (len(missing), list(missing)[:5]))
    if unexpected:
        print('[msvqvae_from_ckpt] 多余 key %d 个: %s' % (len(unexpected), list(unexpected)[:5]))
    return m, ck


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    print('=' * 76)
    print('多尺度残差量化 自检')
    print('=' * 76)

    n_codes, C, H, W = 1024, 64, 16, 16
    cb = nn.Embedding(n_codes, C)
    nn.init.normal_(cb.weight, std=0.5)
    cb.weight.requires_grad_(False)
    ms = MultiScaleResidualVQ(cb, patch_nums=(1, 2, 4, 8, 16))

    print()
    print('[1] 形状与长度')
    print('  patch_nums =', ms.patch_nums)
    print('  各尺度 token 数:', [pn * pn for pn in ms.patch_nums])
    print('  总 token 数: %d  (单尺度是 %d, 多 %.0f%%)'
          % (ms.total, H * W, 100 * (ms.total / (H * W) - 1)))
    print('  begin_ends:', ms.begin_ends)

    f = torch.randn(2, C, H, W)
    ms.eval()
    idxs, f_hat, vq_loss = ms.f_to_idx(f)
    print()
    print('[2] 编码/解码往返')
    print('  idxs 形状:', [tuple(t.shape) for t in idxs])
    print('  f_hat 形状:', tuple(f_hat.shape), ' vq_loss %.4f' % float(vq_loss))
    idxs2, f_hat2, _ = ms.f_to_idx(f)
    print('  两次编码 idx 是否一致:', all(bool((a == b).all()) for a, b in zip(idxs, idxs2)))
    fh3 = ms.idx_to_fhat(idxs, H, W)
    d = (f_hat - fh3).abs().max().item()
    print('  f_to_idx 的 f_hat vs idx_to_fhat 最大差: %.3e  %s'
          % (d, 'OK' if d < 1e-3 else 'WARN（STE 与 no_grad 路径可能有细微差异）'))

    print()
    print('[3] 残差是否真的逐级递减（关键性质）')
    f_hat = torch.zeros_like(f)
    for i, pn in enumerate(ms.patch_nums):
        f_k = F.interpolate(f, size=(pn, pn), mode='bilinear', align_corners=False)
        fh_k = F.interpolate(f_hat, size=(pn, pn), mode='bilinear', align_corners=False)
        r_k = f_k - fh_k
        idx_k = ms._match(r_k / ms._scale_of(i))
        q_k = ms._gather(idx_k, r_k.shape) * ms._scale_of(i)
        f_hat = f_hat + F.interpolate(q_k, size=(H, W), mode='bilinear', align_corners=False)
        err = (f - f_hat).pow(2).mean().sqrt().item()
        rq_err = (r_k - q_k).pow(2).mean().sqrt().item()
        print('  尺度 %d (pn=%2d, %3d tok): 重建 RMSE %.4f   残差量化误差 %.4f'
              % (i, pn, pn * pn, err, rq_err))

    print()
    print('[4] STE 是否真的通了梯度（v1 就是死在这里）')
    cb2 = nn.Embedding(n_codes, C)
    nn.init.normal_(cb2.weight, std=0.5)
    cb2.weight.requires_grad_(False)
    ms2 = MultiScaleResidualVQ(cb2, patch_nums=(1, 2, 4, 8, 16))
    ms2.train()
    f_in = torch.randn(2, C, H, W, requires_grad=True)
    _, fh, _ = ms2.f_to_idx(f_in)
    fh.pow(2).mean().backward()
    g = f_in.grad
    print('  d(f_hat)/d(f) 非零元素占比: %.1f%%   梯度绝对值均值 %.3e'
          % (100.0 * (g.abs() > 0).float().mean().item(), g.abs().mean().item()))
    print('  => 若为 0% 则 STE 断了（v1 的 bug）；接近 100%% 才正常')

    print()
    print('[5] 序列打包/解包')
    seq = ms.pack(idxs)
    print('  打包:', tuple(seq.shape))
    back = ms.unpack(seq)
    print('  解包一致:', all(bool((a == b).all()) for a, b in zip(idxs, back)))

    print()
    print('[6] block mask（VAR 的 lvl(i) >= lvl(j)）')
    m = ms.block_mask()
    print('  mask 形状:', tuple(m.shape))
    a, b = ms.begin_ends[2]
    print('  尺度 2 内（16 个 token）是否全可见:', bool(m[a:b, a:b].all()))
    lo = ms.begin_ends[0][0]
    hi = ms.begin_ends[4][0]
    print('  高尺度(位置 %d) 能看到低尺度(位置 %d):' % (hi, lo), bool(m[hi, lo]))
    print('  低尺度(位置 %d) 能看到高尺度(位置 %d):' % (lo, hi), bool(m[lo, hi]), '（应为 False）')

    print()
    print('[7] 自回归步数')
    print('  单尺度 causal: %d 步' % (H * W))
    print('  多尺度 block:  %d 步（每步一个尺度）' % ms.n_scale)
    print('  => 减少 %.0f×' % (H * W / ms.n_scale))
    print()
    print('=' * 76)
    print('DONE')
