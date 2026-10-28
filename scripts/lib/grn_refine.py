# -*- coding: utf-8 -*-
"""grn_refine.py — GRN 风格的 global refinement（arXiv 2604.13030）

论文机制（Eq.4-7）：
    F_t = S_t · Y_t  ⊕  (1−S_t) · Y_rand          (4)
    p(Y_{t+1}) = Φ(F_t, cond)                      (5)
    S_t = RandLike(Y_t) < l_t                      (6)
    L = −E[(1/N) Σ_i log p(y_i | F_t, cond)]       (7)  普通 CE

  训练：l_t 在 [0,1] 均匀采样；Y_rand ~ Uniform；位置沿 spatial/channel 随机选
  推理：l_t 从 0% 单调升到 100%，【每步全量重算所有 token，允许擦除已画错的】

论文两个反向教学（务必遵守，都是硬数字）：
  ❌ 冻结已生成 token（MaskGIT 式）→ FID 【3.63 vs 185.62】
  ❌ 按置信度选 token 更新        → FID 【3.63 vs 10.64】
  ⇒ 必须"全量重算 + 允许擦除"，任何"冻结/按置信度挑"都会崩

为什么这对我们可能有救：
  我们的实测（2026-09-18）：光栅序 causal 下 T=256 的逐位置 top-1 单调下降
  （11.5% → 5.7%），且 hier 的生成 token 相邻同率 42~54%（真实 4.47%）。
  根因是"单向因果 + 位置顺序"破坏了 2D 结构。
  GRN 的 refinement 让【所有位置平等、双向、可反复修改】—— 从根本上绕开这两个问题。
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import torch
import torch.nn.functional


def grn_make_input(y_true, vocab, l=None, generator=None):
    """构造 GRN 训练输入 F_t = S·Y_true ⊕ (1−S)·Y_rand

    y_true: (B, T) int64
    l: 保留比例；None 表示从 U[0,1] 采样（论文 Eq.6）
    返回 (x_in, l, S)
    """
    B, T = y_true.shape
    dev = y_true.device
    if l is None:
        l = float(torch.rand(1, generator=generator).item())
    S = (torch.rand(B, T, device=dev, generator=generator) < l)
    y_rand = torch.randint(0, vocab, (B, T), device=dev, generator=generator)
    x_in = torch.where(S, y_true, y_rand)
    return x_in, l, S


def grn_schedule(steps, t0=5, alpha=50.0, T_min=8, T_max=None, shape='linear'):
    """推理时的 l_t 日程：从 0 单调升到 1。

    论文的 Complexity 调度器（Eq.9）需要逐步算熵，CPU 上开销大；
    这里提供 linear / cosine / step 三种简化版（论文实测 Complexity 比
    Linear 好，但 Linear 也比"固定步数"可控）。
    """
    if T_max is None:
        T_max = steps
    ts = torch.arange(steps, dtype=torch.float32)
    if shape == 'linear':
        l = ts / max(1, steps - 1)
    elif shape == 'cosine':
        l = 0.5 - 0.5 * torch.cos(torch.pi * ts / max(1, steps - 1))
    elif shape == 'step':
        # 前 1/3 快速升到 0.5，后 2/3 慢升（对应论文"中间步掩码收敛"的观察）
        l = torch.where(ts < steps / 3,
                        ts / (steps / 3) * 0.5,
                        0.5 + 0.5 * (ts - steps / 3) / max(1, steps - 1 - steps / 3))
    else:
        raise ValueError(shape)
    return l.clamp(0.0, 1.0)


@torch.no_grad()
def grn_sample(model, batch, T, vocab, steps=20, shape='step',
               temperature=1.0, top_k=100, generator=None, device='cpu'):
    """GRN 风格采样：迭代 refinement，【全量重算 + 允许擦除】。

    ★ 2026-09-21 修复：原实现用 `g.argmax(-1)` 取值。
      但我们实测过同一个坑（§10.2 第 24 条，MaskGIT）：**argmax 是确定性函数，
      作用在同一个随机起点上会让整个 batch 退化**，并【伪造出"码本坍塌"的假象】
      （当时码本从 98.1% 假摔到 2.9%）。
      GRN 这次实测：argmax 下码本只用 25/1024（2.4%）、熵 26.6%、产出近乎全黑 —— 同一个指纹。
      ⇒ 改为按 top-k 分布采样。`temperature` 参数原先存在但【从未被使用】，也是静默失效。

    model 需提供 forward_logits(tokens) -> (_, gen_logits)
    返回 (B, T) 的 index
    """
    import torch.nn.functional
    dev = torch.device(device)
    B = batch
    # 起点：全随机（论文：l_0 = 0）
    y = torch.randint(0, vocab, (B, T), device=dev, generator=generator)
    l_sched = grn_schedule(steps, shape=shape).to(dev)

    def _pick(g):
        if temperature is None or temperature <= 0:
            return g.argmax(-1)                      # 论文口径（供对照）
        lg = g / float(temperature)
        k = min(int(top_k), lg.shape[-1])
        v, _ = torch.topk(lg, k, dim=-1)
        thr = v[..., -1, None]
        lg = torch.where(lg < thr, torch.full_like(lg, float('-inf')), lg)
        pr = torch.softmax(lg, dim=-1)
        return torch.multinomial(pr.reshape(-1, pr.shape[-1]), 1).reshape(pr.shape[:-1])

    for t in range(steps):
        lt = float(l_sched[t].item())
        _, g = model.forward_logits(y)
        g = g[0] if isinstance(g, (tuple, list)) else g
        pred = _pick(g)
        # 【关键】全量重算 + 允许擦除：按 l_t 决定保留预测还是换回随机
        S = (torch.rand(B, T, device=dev, generator=generator) < lt)
        y_rand = torch.randint(0, vocab, (B, T), device=dev, generator=generator)
        y = torch.where(S, pred, y_rand)
    # 最后一步全量用预测
    _, g = model.forward_logits(y)
    g = g[0] if isinstance(g, (tuple, list)) else g
    return _pick(g)


if __name__ == '__main__':
    torch.manual_seed(0)
    print('=' * 70)
    print('GRN refinement 自检')
    print('=' * 70)

    V, B, T = 1024, 4, 256
    y = torch.randint(0, V, (B, T))

    # ① 输入构造
    for l in (0.0, 0.3, 0.7, 1.0):
        x, lv, S = grn_make_input(y, V, l=l)
        keep = (x == y).float().mean().item()
        print('  l=%.1f → 实际保留比例 %.3f  (应≈%.1f)  l 返回 %.1f'
              % (l, keep, l, lv))

    # ② l 从 U[0,1] 采样
    ls = [grn_make_input(y, V)[1] for _ in range(200)]
    ls = torch.tensor(ls)
    print('  l 采样: 均值 %.3f (应≈0.5)  范围 [%.3f, %.3f]'
          % (ls.mean(), ls.min(), ls.max()))

    # ③ 日程
    print()
    print('  推理日程 l_t（steps=20）:')
    for shape in ('linear', 'cosine', 'step'):
        s = grn_schedule(20, shape=shape)
        print('    %-8s %s' % (shape, ' '.join('%.2f' % v for v in s[::4].tolist())))

    # ④ 用一个小模型跑一遍采样，确认形状与合法性
    print()
    import small_image_model_v2
    cfg = M.SmallImageConfigV2(
        vocab_size=V, d_model=128, n_layer=2, n_head=4, n_kv_head=2,
        ffn_dim=512, total_tokens=T, grid=16, think_tokens=8,
        text_dim=256, ada_dim=64, cond_mode='none', cond_dropout=0.1,
        pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
        hier=False, loop_start=1, loop_end=2, loop_times=1, use_bos=False,
        mask_mode='block')
    m = M.SmallARImageModelV2(cfg)
    m.eval()
    out = grn_sample(m, 2, T, V, steps=4, device='cpu')
    print('  采样输出 %s  范围 [%d, %d]  %s'
          % (tuple(out.shape), out.min(), out.max(),
             'OK' if 0 <= out.min() and out.max() < V else 'FAIL'))
    print()
    print('DONE')
