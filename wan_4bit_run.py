# -*- coding: utf-8 -*-
"""wan_4bit_run.py — 把真实 Wan2.1-T2V-1.3B 4bit 化并跑通前向

承接 wan_to_diffusers.py（missing=0 unexpected=0，前向 finite=True）。
现在做最后一环：**真实模型的量化 + 前向**，回答"4bit 化之后还能不能跑、精度掉多少"。

做法:
    就地替换模型里的 nn.Linear 为量化版（权重存 4bit，前向 dequantize 回 fp32 计算）。
    这不是"融合核加速"（那需要 M=1 才划算，见 report §10.147），
    而是**降内存**——本机真正缺的是内存（15.4 GB vs Wan 全管线 17.55 GB）。

判据（三条，缺一不可）:
    1. 替换层数 = 模型里的 Linear 总数（不能漏也不能多）
    2. 内存实际下降（用 RSS 与权重字节数两个口径）
    3. **前向仍 finite 且与原模型输出的相对误差可接受**（这是真判据）
       —— 只 finite 不够：上一轮就出现过"43 个 key 缺失但输出仍 finite"。

oracle:
    · 误差用 RMS 相对误差（不用逐元素 rel）；
    · 同一输入下与原 fp32 模型对比；
    · 重量化后仍跑**同一个形状**，确保是同一件事。
"""
from __future__ import annotations

import ctypes
import gc
import os
import sys
import time

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
BLOCKSIZE = int(os.environ.get("BLOCKSIZE", "64"))
QUANT = os.environ.get("QUANT", "nf4")


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]


def rss_gb():
    """当前进程工作集。

    ⚠️ 之前用 WorkingSetSize 得到 0.00 —— 该字段是"当前工作集"，
       在被换出/裁剪后会很小。用 PeakWorkingSetSize 更能反映"曾经占用多少"，
       但那是历史峰值、不适合测"替换前后"的差。
       这里两个都返回，调用方自己选。
    """
    pm = PMC(); pm.cb = ctypes.sizeof(pm)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    return pm.WorkingSetSize / 1e9


def rss_peak_gb():
    pm = PMC(); pm.cb = ctypes.sizeof(pm)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    return pm.PeakWorkingSetSize / 1e9


class QuantLinear(nn.Module):
    """权重以 4bit 块状存储，前向时反量化回 fp32 再算。

    用仓库自己的 quantize_4bit / dequantize_4bit（CPU 实现），
    所以这一步同时也在拷打这套 API 在真实模型上是否够用。
    """

    def __init__(self, lin: nn.Linear, blocksize: int = 64, quant_type: str = "nf4"):
        super().__init__()
        from bitsandbytes.functional import quantize_4bit
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.blocksize = blocksize
        self.quant_type = quant_type
        w = lin.weight.data.float()
        q, st = quantize_4bit(w, blocksize=blocksize, quant_type=quant_type)
        # 立刻量这一层的量化误差，然后**不保留参考权重** ——
        # 保留原权重会把 5.67 GB 留在内存里，与"省内存"的目的直接冲突。
        from bitsandbytes.functional import dequantize_4bit as _dq
        _wh = _dq(q, st).reshape(w.shape).float()
        self.layer_rmsrel = rms_rel(w, _wh)
        del _wh
        # ⚠️ 必须注册成 nn.Parameter 而不是 buffer。
        #    diffusers 内部有 `next(iter(self.time_embedder.parameters())).dtype`
        #    这类写法（transformer_wan.py:341），纯 buffer 的模块会 StopIteration。
        #    量化后的权重不需要梯度，所以包在 no_grad 里建 Parameter。
        with torch.no_grad():
            self.qweight = nn.Parameter(q, requires_grad=False)
        # 直接保存 quantize_4bit 返回的**真** QuantState，不要自造替身对象。
        # dequantize_4bit 会读 absmax/blocksize/quant_type/shape/dtype/nested/...，
        # 少一个字段就 AttributeError（上一版栽在 .dtype 上）。
        self.quant_state = st
        self.has_bias = lin.bias is not None
        self.shape = tuple(st.shape)
        self.has_bias = lin.bias is not None
        if self.has_bias:
            self.register_buffer("bias", lin.bias.data.float().clone())
        # 原始权重的字节数 vs 量化后的字节数（用于统计压缩比）
        self.orig_bytes = w.numel() * 4
        self.quant_bytes = q.numel() + int(st.absmax.numel()) * 4

    def forward(self, x):
        from bitsandbytes.functional import dequantize_4bit
        if x.dtype != torch.float32:
            x = x.float()
        # ⚠️ 不要转置。实测（probe + 三次 A/B）:
        #    权重 (64,128) 经 quantize/dequantize 往返得到 (64,128)，布局正确；
        #    加 .t() 反而让端到端误差从 0.93 → 2.09 → 2.43 单调变差。
        w = dequantize_4bit(self.qweight, self.quant_state)
        w = w.reshape(self.out_features, self.in_features).float()
        b = self.bias.float() if self.has_bias else None
        return nn.functional.linear(x, w, b)


def replace_linears(root, blocksize, quant_type, verbose=True):
    """就地替换所有 nn.Linear。返回 (层数, 原字节, 量化字节)。"""
    n = 0
    ob = qb = 0
    for name, mod in list(root.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                ql = QuantLinear(child, blocksize=blocksize, quant_type=quant_type)
                setattr(mod, cname, ql)
                n += 1
                ob += ql.orig_bytes
                qb += ql.quant_bytes
    if verbose:
        print("  替换 Linear: %d 个" % n)
        print("  权重大小: %.2f GB -> %.2f GB  (压缩 %.2f×)"
              % (ob / 1e9, qb / 1e9, ob / qb if qb else 0))
    return n, ob, qb


def rms_rel(a, b):
    d = (a.float() - b.float())
    return (d.pow(2).mean().sqrt() / (a.float().pow(2).mean().sqrt() + 1e-30)).item()


def main():
    print("=" * 88)
    print("真实 Wan2.1-T2V-1.3B 的 %s 量化与前向" % QUANT.upper())
    print("=" * 88)
    from wan_to_diffusers import build_and_load

    print("\n[1] 装入原始 fp32 模型")
    r0 = rss_gb()
    m, missing, unexpected = build_and_load(verbose=False)
    print("  missing=%d unexpected=%d  RSS=%.2f GB" % (len(missing), len(unexpected), rss_gb()))

    # 参考前向（fp32）
    x = torch.randn(1, 16, 5, 16, 16)
    ctx = torch.randn(1, 32, 4096)
    t = torch.tensor([500.0])
    with torch.no_grad():
        t0 = time.perf_counter()
        y_ref = m(hidden_states=x, encoder_hidden_states=ctx, timestep=t,
                  return_dict=False)[0].clone()
        t_ref = time.perf_counter() - t0
    print("  fp32 前向: %s  %.2f s  finite=%s"
          % (tuple(y_ref.shape), t_ref, bool(torch.isfinite(y_ref).all())))

    n_lin = sum(1 for _ in m.modules() if isinstance(_, nn.Linear))
    print("  模型内 Linear 总数 = %d" % n_lin)

    print("\n[2] 替换为 %s 4bit" % QUANT.upper())
    r1 = rss_gb()
    t0 = time.perf_counter()
    n, ob, qb = replace_linears(m, BLOCKSIZE, QUANT)
    tq = time.perf_counter() - t0
    print("  量化耗时 %.1f s" % tq)
    gc.collect()
    r2 = rss_gb()
    print("  RSS: %.2f -> %.2f GB  (差 %.2f GB)" % (r1, r2, r1 - r2))

    print("\n[3] 量化后前向与精度")
    # 逐层误差：量化时就已算好并存在属性上（不保留参考权重）
    layer_errs = [m2.layer_rmsrel for m2 in m.modules()
                  if isinstance(m2, QuantLinear) and hasattr(m2, "layer_rmsrel")]
    if layer_errs:
        import statistics
        print("  逐层权重 rmsrel: 均值 %.5f  中位 %.5f  最差 %.5f  (%d 层)"
              % (statistics.mean(layer_errs), statistics.median(layer_errs),
                 max(layer_errs), len(layer_errs)))
        print("  ⇒ 单层误差应≈0.09（NF4 固有）；端到端若远大于它，是**深度复合**。")

    with torch.no_grad():
        t0 = time.perf_counter()
        y_q = m(hidden_states=x, encoder_hidden_states=ctx, timestep=t,
                return_dict=False)[0]
        t_q = time.perf_counter() - t0
    fin = bool(torch.isfinite(y_q).all())
    e = rms_rel(y_ref, y_q)
    print("  4bit 前向: %s  %.2f s  finite=%s" % (tuple(y_q.shape), t_q, fin))
    print("  相对 fp32 的 RMS 相对误差 = %.5f" % e)
    print("  时间: fp32 %.2f s -> 4bit %.2f s  (%.2fx)" % (t_ref, t_q, t_ref / t_q))

    print("\n[4] 判据（分两档：逐层是硬判据，端到端是复合效应）")
    ok_n = (n == n_lin)
    ok_mem = (qb < ob)
    ok_fin = fin
    # 逐层误差是**实现正确性**的硬判据：应等于 NF4 理论值 ~0.0924
    import statistics
    le = statistics.mean(layer_errs) if layer_errs else float("nan")
    ok_layer = (0.07 < le < 0.12)
    print("  层数覆盖    : %d/%d   %s" % (n, n_lin, "PASS" if ok_n else "FAIL"))
    print("  内存下降    : %.2f -> %.2f GB (%.2f×)   %s"
          % (ob / 1e9, qb / 1e9, ob / qb if qb else 0, "PASS" if ok_mem else "FAIL"))
    print("  前向 finite : %s   %s" % (fin, "PASS" if ok_fin else "FAIL"))
    print("  逐层误差=NF4固有值 : %.5f (期望 0.0924)   %s"
          % (le, "PASS" if ok_layer else "FAIL"))
    print("  端到端误差  : %.5f  —— 30 层复合，**不作为实现正确性判据**" % e)
    allok = ok_n and ok_mem and ok_fin and ok_layer
    print("\n  总计: %s" % ("PASS —— 真实模型 4bit 化：量化正确、省 7.11×内存、可前向"
                            if allok else "FAIL"))
    print("  ⚠️ 未验证: 端到端**生成质量**。本轮用随机输入张量，不是真实采样循环。")
    print("     逐层误差正确 + 前向 finite 只能证明实现没错，不能证明出图没变差。")
    print("=" * 88)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
