# -*- coding: utf-8 -*-
"""looped_dit.py — 循环扩散 Transformer（LoopedMDM 架构的 DiT 实现）

设计目标（对照实验）：
  - LoopedDiT: n_unique 个独特块，其中 [loop_start, loop_end] 闭区间内的块
    重复 n_loops 次，总有效块数 = loop_start + 窗口宽×n_loops + 尾部块数
  - 参数量与 UNet2D-128/256/256 对齐（≈25.9M）
  - 全部为确定性算子（fwd/bwd 可复现），纯 CPU 可跑

⚠️ 接口与 diffusers 的 UNet2DModel 一致：forward(x, t) 里 x 是 (B, C, G, G)，
   返回同样形状，所以可以直接替换进 train_latent_diffusion.py。

⚠️ 注意签名默认值是 d=512，而"≈25.9M"这个设计目标对应的是 **d=384**：
   每块 = 22d² + 6d（LayerNorm 无 affine，不计参）
     d=384 → 3.246M/块 ×8 = 25.97M   ✅ 对齐
     d=512 → 5.770M/块 ×8 = 46.16M   ❌ 1.78 倍
   所以要复现文档里的规模，必须显式传 d=384。

性能修订（2026-09-23）：
  1. rope_2d 的频率/meshgrid/cos/sin 原本每次调用都重算 —— 这些只依赖
     (grid, half, device, dtype)，与数据无关。改为带缓存的常量表。
  2. eff_depth 原为 n_unique + n_loops，在 n_loops=1 时报 9 而实际只执行 8 块。
     改为直接等于 block_executions(...)。
  3. gate / loop-update 探针原为【每次前向无条件执行】，其中
     - gate 探针重复调用各块的 self.ada(t_emb_loop)（DiTBlock 内部已算过一次）
     - .item() 每轮强制设备同步
     - upd.detach().pow(2).mean().sqrt() 多分配两个张量
     改为 opt-in（self.probe = True 才跑），并把范数换成 .norm()。
     默认路径不再有任何同步点和额外探查开销。
"""
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
import os
import torch
import torch.nn
import torch.nn.functional


# ---------------------------------------------------------------------------
# 2D RoPE 的常量表缓存
# ---------------------------------------------------------------------------
# 表只依赖 (grid, half, device, dtype)，与 q/k 数据无关。
# 缓存的张量很小（T × half/2 个 float），但与设备绑定，所以限制条目数，
# 避免在 DML 这类显存受限的设备上无限攒表。
_ROPE_CACHE = {}
_ROPE_CACHE_MAX = 8


def _rope_tables(grid, half, device, dtype):
    """返回 (cos_y, sin_y, cos_x, sin_x)，每个形状 (T, half//2)。"""
    key = (grid, half, str(device), str(dtype))
    tab = _ROPE_CACHE.get(key)
    if tab is not None:
        return tab

    ys, xs = torch.meshgrid(
        torch.arange(grid, device=device), torch.arange(grid, device=device), indexing="ij"
    )
    y = ys.reshape(-1).to(dtype)
    x = xs.reshape(-1).to(dtype)

    # 频率表：half//2 个频率（每组 2 维共用一个频率）
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, 2, device=device).to(dtype) / half
    )
    ang_y = y[:, None] * freqs[None, :]   # (T, half/2)
    ang_x = x[:, None] * freqs[None, :]

    tab = (ang_y.cos(), ang_y.sin(), ang_x.cos(), ang_x.sin())
    if len(_ROPE_CACHE) >= _ROPE_CACHE_MAX:
        _ROPE_CACHE.clear()
    _ROPE_CACHE[key] = tab
    return tab


def _rope_tables_full(grid, half, device, dtype):
    """把 cos/sin 预先展开成 (1,1,T,half) 的【完整布局】，每个频率重复两次。

    为什么这么做：`rot()` 里往 `out[..., 0::2]` / `out[..., 1::2]` 写是【跨步散写】，
    加上每次都 torch.empty 分配、最后还要 torch.cat —— 全是小算子开销。
    实测 rope_2d 占扩散整步的 8.3%，而 q/k 各 12.6MB 的纯带宽只要约 1.7ms/block，
    差 90 倍，说明瓶颈就是这些开销而不是访存。

    预先把表摆成重复两次的完整形状后，旋转退化成
        out = t * c_full + rotate_half(t) * s_full
    只剩 reshape / 两次 gather / stack / flatten / 2乘1加。

    实测（bench_rope.py，q,k=(8,8,1024,48) 的 fwd+bwd）：
        原始 280.60 ms -> 全表 104.05 ms  = 0.37x，且输出【逐位相同】(max_abs=0)
    """
    key = (grid, half, str(device), str(dtype), 'full')
    tab = _ROPE_CACHE.get(key)
    if tab is not None:
        return tab
    cy, sy, cx, sx = _rope_tables(grid, half, device, dtype)
    # 前一半用 y 频率、后一半用 x 频率，各自每 2 维一组共用同一频率
    c = torch.cat([cy, cx], dim=-1).repeat_interleave(2, dim=-1)[None, None]
    s = torch.cat([sy, sx], dim=-1).repeat_interleave(2, dim=-1)[None, None]
    tab = (c, s)
    if len(_ROPE_CACHE) >= _ROPE_CACHE_MAX:
        _ROPE_CACHE.clear()
    _ROPE_CACHE[key] = tab
    return tab


def _rotate_half(x):
    """(-x1, x0, -x3, x2, ...)：相邻两维互换并把第一个取负。"""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack((-x1, x0), dim=-1).flatten(-2)


_ROPE_FUSED = os.environ.get('CPU_FORGE_ROPE', 'fused').lower() != 'legacy'


def rope_2d(q, k, grid):
    """2D 旋转位置编码（RoPE）。
    输入 q,k: (B, H, T, D)，T = grid*grid 为空间展平序列。
    设计：d_head 前一半用 y 坐标频率旋转，后一半用 x 坐标频率旋转，每 2 维一组。
    """
    B, H, T, D = q.shape
    half = D // 2
    assert half % 2 == 0 and T == grid * grid, f"D={D} half={half} T={T} grid={grid}"

    if _ROPE_FUSED:
        c, s = _rope_tables_full(grid, half, q.device, q.dtype)
        return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s

    cy, sy, cx, sx = _rope_tables(grid, half, q.device, q.dtype)

    def rot(t, c, s):
        # t: (B, H, T, half) -> 每 2 维一组旋转
        t = t.reshape(B, H, T, half // 2, 2)
        t0, t1 = t[..., 0], t[..., 1]
        out = torch.empty(B, H, T, half, dtype=t.dtype, device=t.device)
        out[..., 0::2] = t0 * c - t1 * s
        out[..., 1::2] = t0 * s + t1 * c
        return out

    qy, qx = rot(q[..., :half], cy, sy), rot(q[..., half:], cx, sx)
    ky, kx = rot(k[..., :half], cy, sy), rot(k[..., half:], cx, sx)
    return torch.cat([qy, qx], dim=-1), torch.cat([ky, kx], dim=-1)


_AFFINE_CACHE = {}
# 非 CPU 设备上用基础算子重写 LayerNorm（绕开 torch-directml 的原生反向 bug）
_DML_SAFE_NORM = os.environ.get('CPU_FORGE_SAFE_NORM', '1') != '0'


def _affine_ones_zeros(shape, ref):
    """给 LayerNorm 造显式的 ones/zeros（按 设备+精度 缓存）。

    ★ 为什么需要：DiTBlock 用的是 `nn.LayerNorm(d, elementwise_affine=False)`，
      于是 F.layer_norm 收到 weight=None, bias=None。**torch-directml 0.2.5 在
      `NativeLayerNormBackward0` 走这条 None 路径时会崩**，报
      `tensor does not have a device`（前向 OK、反向炸），整个训练没法跑。

      传显式的 ones/zeros 就绕开了这条路径（实测单算子 fwd+bwd 正常），
      数值完全相同、state_dict 不变。

    ★ 只在【非 CPU】设备上走这条路 —— CPU 上原实现又快又好，
      而且正在排队的训练任务不允许被无谓地动。
    """
    key = (tuple(shape), str(ref.device), str(ref.dtype))
    v = _AFFINE_CACHE.get(key)
    if v is None:
        w = torch.ones(*shape, device=ref.device, dtype=ref.dtype)
        b = torch.zeros(*shape, device=ref.device, dtype=ref.dtype)
        v = (w, b)
        _AFFINE_CACHE[key] = v
    return v


def _layernorm_none_affine(mod, x):
    """elementwise_affine=False 的 nn.LayerNorm，在非 CPU 设备上绕开原生算子的 bug。

    ★ 为什么不能直接用：torch-directml 0.2.5 的 `NativeLayerNormBackward0`
      在这个形状/配置下**第一次反向能过、第二次就崩**（裸 RuntimeError）。
      实测（每个算子独立进程、连续 8 次反向）：
          Linear 8/8 ✓   mul 8/8 ✓   hand-RMSNorm 8/8 ✓   LayerNorm 1/8 ✗
      它一崩还会污染同进程后续的所有测试 —— 一开始看着像"全都坏了"，
      分开跑才定位到只有它。
      这跟早先 UNet+GroupNorm 那次的根因是同一类：**DML 的归一化反向不可靠**。

    ⇒ 用基础算子重算同样的数学（mean/sub/pow/rsqrt/mul），那条路 8/8 通过。
      数值等价（只差 fp32 结合律），state_dict 不变。

    CPU 上【原样调用】—— 那边又快又好，而且正在排队的训练任务不允许被无谓地动。
    """
    if x.device.type == 'cpu':
        return mod(x)
    import torch.nn.functional
    w, b = _affine_ones_zeros(mod.normalized_shape, x)
    if not _DML_SAFE_NORM:
        return _F.layer_norm(x, mod.normalized_shape, w, b, mod.eps)
    # 用基础算子重写 LayerNorm：与 F.layer_norm 数学相同
    mu = x.mean(dim=-1, keepdim=True)
    d = x - mu
    y = d * torch.rsqrt(d.pow(2).mean(dim=-1, keepdim=True) + mod.eps)
    return y * w + b


class TimestepEmbedder(nn.Module):
    """正弦时间嵌入 + MLP（MLP hidden = 4*d，SiLU，无输出 norm）"""

    def __init__(self, d):
        super().__init__()
        self.d = d
        half = d // 2
        self.register_buffer("t_freq", torch.exp(-math.log(10000.0) * torch.arange(0, half, 2).float() / half))
        self.mlp = nn.Sequential(
            nn.Linear(half, 4 * d, bias=True),
            nn.SiLU(),
            nn.Linear(4 * d, d, bias=True),
        )

    def forward(self, t):
        a = t[:, None] * self.t_freq[None, :]  # (B, half)
        emb = torch.cat([a.sin(), a.cos()], dim=-1)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    """DiT 块：adaLN-Zero 条件化 + 2D RoPE 自注意力 + SwiGLU FFN。所有 Linear 无 bias。

    cond_dim > 0 时启用文本条件：ada 由时间嵌入和条件嵌入【相加】得到。
    cond 分支零初始化 ⇒ 初始时等价于无条件，不会破坏已有的 adaLN-Zero 结构。
    """

    def __init__(self, d, n_heads, grid, mlp_ratio=4.0, cond_dim=0):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.grid = grid

        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ada = nn.Linear(d, 6 * d, bias=True)  # adaLN-Zero: scale/shift/gate × 2
        self.ada_c = nn.Linear(cond_dim, 6 * d, bias=True) if cond_dim > 0 else None
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.w1 = nn.Linear(d, int(mlp_ratio * d), bias=False)
        self.w2 = nn.Linear(d, int(mlp_ratio * d), bias=False)
        self.w3 = nn.Linear(int(mlp_ratio * d), d, bias=False)

    def forward(self, x, t_emb, c=None, probe=False):
        B, T, _ = x.shape
        ada = self.ada(t_emb)
        if self.ada_c is not None and c is not None:
            ada = ada + self.ada_c(c)
        scale1, shift1, gate1, scale2, shift2, gate2 = ada.chunk(6, dim=-1)

        h = _layernorm_none_affine(self.norm1, x) * (1 + scale1[:, None]) + shift1[:, None]
        q, k, v = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head).unbind(dim=2)
        q, k = rope_2d(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), self.grid)
        v = v.permute(0, 2, 1, 3)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, T, self.d)
        x = x + gate1[:, None] * self.proj(a)

        h = _layernorm_none_affine(self.norm2, x) * (1 + scale2[:, None]) + shift2[:, None]
        h = self.w3(F.silu(self.w1(h)) * self.w2(h))
        x = x + gate2[:, None] * h
        if probe:
            # 复用上面已经算好的 ada，不重复做一遍 matmul；也不在这里 .item()
            self._last_gate = ada[..., [2, 5]].abs().mean()
        return x


class LoopedDiT(nn.Module):
    """
    Looped-DiT 主模型。
    参数:
      in_ch: 输入通道数（latent 为 64；像素测试为 3）
      latent_size: 空间边长（latent 为 16；像素测试为 32）
      d / n_heads: 模型宽度 / 头数
      n_unique: 独特块总数（含循环窗口内的块）
      loop_start / loop_end: 循环窗口 [loop_start, loop_end] 闭区间，窗口内块循环 n_loops 次
      n_loops: 循环次数（第 2..n_loops 次为共享权重重复执行）
    有效深度 = block_executions(n_unique, loop_start, loop_end, n_loops)
    """

    def __init__(self, in_ch=64, latent_size=16, d=512, n_heads=8,
                 n_unique=8, loop_start=2, loop_end=3, n_loops=1,
                 loop_t_emb=False, loop_damp="1/n", cond_dim=0):
        # loop_damp: 循环窗口残差缩放，"1/n"=1/n_loops（spec，Seed 系结论）；
        #            也可传 "1.0"/"0.5" 等（阶段二阻尼消融臂）。
        # cond_dim:  文本条件维度（本项目 CLIP 文本嵌入是 512）。0 = 无条件，
        #            此时行为与加入条件化之前【完全一致】（逐位相同）。
        super().__init__()
        assert 0 <= loop_start <= loop_end < n_unique
        self.in_ch = in_ch
        self.latent_size = latent_size
        self.grid = latent_size
        self.d = d
        self.n_heads = n_heads
        self.n_unique = n_unique
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.n_loops = n_loops
        self.loop_t_emb = loop_t_emb
        self.loop_damp = loop_damp
        self.cond_dim = cond_dim
        # 真实执行块数（原为 n_unique + n_loops，n_loops=1 时错报 9，实际 8）
        self.eff_depth = block_executions(n_unique, loop_start, loop_end, n_loops)
        self.probe = False            # 探针总开关，默认关（原为无条件执行）
        self.last_loop_updates = []   # 探针：每轮循环窗口的更新范数 ||h-h_prev||2
        self.last_gate_by_round = []  # 探针：每轮循环窗口的 gate1/gate2 绝对值均值

        self.t_embed = TimestepEmbedder(d)
        # 抄《On Expressive Power of Looped Transformers》(2410.01405) 的 timestep-encoding 增强：
        # 每轮循环一个可学习嵌入，加到 t_emb 上 ⇒ 各轮循环的 adaLN 缩放/偏移不同，
        # 等效于"循环轮次"也作为条件（Ouro 的 loop-aware 深度分配同思路）。
        if loop_t_emb:
            self.loop_emb = nn.Embedding(n_loops + 1, d)
        self.patch = nn.Linear(in_ch, d, bias=True)
        self.layers = nn.ModuleList([DiTBlock(d, n_heads, latent_size, cond_dim=cond_dim)
                                     for _ in range(n_unique)])
        self.out_ada = nn.Linear(d, 2 * d, bias=True)
        self.out_ada_c = nn.Linear(cond_dim, 2 * d, bias=True) if cond_dim > 0 else None
        self.out = nn.Linear(d, in_ch, bias=True)
        self.apply(self._init_weights)
        # adaLN-Zero：调制层零初始化（spec：gate 初始=0 ⇒ block 初始恒等映射）
        # _init_weights 的 xavier 会覆盖它们，此处显式再置零（含每个 DiTBlock 的 ada）
        for blk in self.layers:
            nn.init.zeros_(blk.ada.weight)
            nn.init.zeros_(blk.ada.bias)
        nn.init.zeros_(self.out_ada.weight)
        nn.init.zeros_(self.out_ada.bias)
        # 条件分支同样零初始化 ⇒ 初始时模型等价于无条件模型，
        # 条件信号在训练中自己长起来，不会在开头干扰已有的初始化。
        if cond_dim > 0:
            for blk in self.layers:
                nn.init.zeros_(blk.ada_c.weight)
                nn.init.zeros_(blk.ada_c.bias)
            nn.init.zeros_(self.out_ada_c.weight)
            nn.init.zeros_(self.out_ada_c.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, t, c=None):
        B, C, G, G = x.shape
        T = G * G
        t_emb = self.t_embed(t)  # (B, d)
        probe = self.probe

        h = self.patch(x.flatten(2).transpose(1, 2))  # (B, T, d)

        for i in range(self.loop_start):
            h = self.layers[i](h, t_emb, c)

        if probe:
            self.last_loop_updates = []
            self.last_gate_by_round = []

        n_exec = max(1, self.n_loops)  # 循环窗口执行次数（spec：n_loops 次；n_loops=0 退化为普通 1 次）
        if self.loop_damp == "1/n":
            damp = 1.0 / self.n_loops if self.n_loops > 0 else 1.0
        else:
            damp = float(self.loop_damp)

        for r in range(n_exec):  # 窗口块共享权重重复执行
            if self.loop_t_emb:
                loop_idx = torch.full((t_emb.shape[0],), r, device=t_emb.device, dtype=torch.long)
                t_emb_loop = t_emb + self.loop_emb(loop_idx)
            else:
                t_emb_loop = t_emb
            h_prev = h
            for i in range(self.loop_start, self.loop_end + 1):
                h = self.layers[i](h, t_emb_loop, c, probe=probe)
            if probe:
                # 各块在 forward 里已经把 gate 存进 _last_gate，这里只汇聚
                g_abs = [self.layers[i]._last_gate for i in range(self.loop_start, self.loop_end + 1)]
                self.last_gate_by_round.append(
                    float(torch.stack(g_abs).mean().item())
                )
            if self.n_loops > 0:
                upd = h - h_prev
                if probe:
                    self.last_loop_updates.append(float(upd.detach().norm().item()))
                h = h_prev + damp * upd

        for i in range(self.loop_end + 1, self.n_unique):
            h = self.layers[i](h, t_emb, c)

        out_ada = self.out_ada(t_emb)
        if self.out_ada_c is not None and c is not None:
            out_ada = out_ada + self.out_ada_c(c)
        scale, shift = out_ada.chunk(2, dim=-1)
        h = h * (1 + scale[:, None]) + shift[:, None]
        out = self.out(h).transpose(1, 2).reshape(B, C, G, G)
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# 实际会执行的 block 次数（用于算力预算/有效深度）
def block_executions(n_unique, loop_start, loop_end, n_loops):
    n_exec = max(1, n_loops)
    return loop_start + (loop_end - loop_start + 1) * n_exec + (n_unique - loop_end - 1)


if __name__ == "__main__":
    print("=== 参数量与有效深度（eff_depth 现应与实际块数一致）===")
    for d in (384, 512):
        m = LoopedDiT(d=d, n_unique=8, loop_start=2, loop_end=3, n_loops=1)
        print(f"  d={d}  params={m.n_params()/1e6:.2f}M  "
              f"实际块数={block_executions(8,2,3,1)}  eff_depth={m.eff_depth}")
    m = LoopedDiT(d=384, n_unique=8, loop_start=2, loop_end=3, n_loops=3)
    print(f"  n_loops=3  实际块数={block_executions(8,2,3,3)}  eff_depth={m.eff_depth}")

    print()
    print("=== 探针默认关闭（不产生同步点）===")
    x = torch.randn(2, 64, 16, 16)
    t = torch.rand(2)
    for nl in (1, 3):
        m = LoopedDiT(d=384, in_ch=64, latent_size=16, n_unique=8,
                      loop_start=2, loop_end=3, n_loops=nl)
        y = m(x, t)
        print(f"  n_loops={nl}: out {tuple(y.shape)}  "
              f"默认 probe={m.probe}  gate记录={m.last_gate_by_round}")
        m.probe = True
        y = m(x, t)
        print(f"    probe=True 时: gate={[round(v,6) for v in m.last_gate_by_round]}  "
              f"upd={[round(v,6) for v in m.last_loop_updates]}")
