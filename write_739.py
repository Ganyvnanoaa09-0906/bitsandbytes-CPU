"""Append 7.39: the borrowing plan -- what to copy from CUDA, AMD and Intel, and the order.

The user's direction: stop inventing in isolation, go read what the GPU and accelerator vendors
already solved, and adapt it. Three areas are open and each maps onto a measured bottleneck:

  iGPU LDS        66-68 GB/s against 544 needed (8x) -- and the 300 GFLOPS baseline is not
                  reproducible, so the true ceiling is unknown
  CPU FMA         n=384 gives 371 GFLOPS in L2, n=2048 gives 230 out of it -- a working-set
                  problem, and nobody has tried blocking for L2 residency
  memory          23.7 GB/s measured against 42.7 theoretical, 277 ns latency, saturating at
                  three threads

The plan is measurements first, on an idle machine, then experiments. The grid-16 comparison is
running and will finish in the morning, which is also when the machine goes quiet.

Sources found are recorded with what each is for, so the next round does not have to search
again. Nothing here is measured yet except where it cites an existing measurement.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.39 抄作业计划：CUDA / AMD / Intel 那边有什么现成的

用户的判断：**不要再闭门造车，去看 GPU 与加速器厂商已经解决过的问题，然后搬过来。**
三个方向都还开着，而每一个都对着一个已测出的瓶颈。

#### 7.39.1 三个方向与各自的证据

| 方向 | 已测到的瓶颈 | 未知的是什么 |
|---|---|---|
| **核显 LDS** | **66~68 GB/s，而需要 544（缺口 8 倍）** | **300 GFLOPS 这个基线本身不可复现**（v4 现在给 126.9 且有错）⇒ **真实上限不知道** |
| **CPU FMA** | n=384 在 L2 内 **371 GFLOPS**，n=2048 掉到 **230** | 同一内核只改尺寸，**没人试过为 L2 驻留做分块** |
| **内存带宽** | 实测读 **23.7**（4 线程）/ **27.2**（6 线程）对理论 **42.7**；延迟 **277 ns**；**3 线程就饱和** | 缺的是"在飞请求数"的实际上限 |

#### 7.39.2 搜集到的可抄清单

| 来源 | 抄什么 | 打我们哪里 |
|---|---|---|
| [flash-attention-cpu](https://github.com/amaye15/flash-attention-cpu) | **纯 Rust 的 CPU FlashAttention，明确支持 AVX2，分 v1/v2/v3** | attention 占一步 **18.8%** 而效率未测 |
| [Intel IPEX 优化器融合](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/tutorials/technical_details/optimizer_fusion_cpu.html) | **优化器逐元素操作融合** | **256 token 时优化器占整步 50%**（3.37 s 里 1.68 s 是 NS） |
| [Deep Learning Models on CPUs](https://ar5iv.labs.arxiv.org/html/2206.10034) | **层融合 + 优化器融合**的方法论 | 同上，且给了系统化框架 |
| [torch.compile 的实测](https://github.com/ultralytics/ultralytics/pull/21975/files) · [HF 文档](https://raw.githubusercontent.com/huggingface/transformers/1f8daee0c80cd53a44b1094b79309dd8675d6392/docs/source/en/torch_compile.md) | **Inductor 的算子融合** | **490 次 `aten::mul`** + 调度开销 |
| [AMD ROCm 编程指南](https://rocm-handbook.amd.com/_/downloads/amd-rocm-programming-guide/en/docs-7.14.1/pdf/) | **LDS 与数据移动引擎（DME）** | 核显 LDS 缺口 |
| [AMD IL 规范](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/programmer-references/AMD_Intermediate_Language_IL_Specification_v2.pdf) | **LDS 指令语义（`lds_and_id` 等）** | 判断 66 GB/s 是硬件还是我的微基准错 |
| [DGEMM 向量化微内核与访存感知并行](https://csit.am/2025/proceedings/Workshop/W_6.pdf) | **向量化微内核 + 访存感知分块** | CPU FMA 的 L2 落差分块 |

#### 7.39.3 执行顺序（明天，机器空闲后）

**第一步必须是干净测量，不是实验。** 依据是 7.36：现有数字里有一批是在别的负载下采的，
而所有实验都建立在这些数字上。

| 顺序 | 内容 | 为什么这个顺序 |
|---|---|---|
| **①** | **7.36 的四项重测**（batch 8 profiler · 3000 步基线 · Muon 总墙钟 · 长跑分段耗时） | 地基。**机器必须完全空闲，不许豁免** |
| **②** | **核显真实上限**：用 CLBlast 或重写内核，把 300 GFLOPS 这个数定下来 | 它决定 LDS 缺口是 8 倍还是别的 |
| **③** | **LDS 微基准的独立复现** | 判断 66 GB/s 是硬件事实还是我的测量错 |
| **④** | **memory-level parallelism**：不同独立流数量下的带宽 | 找出在飞请求数的上限 |
| **⑤** | 然后才是实验：torch.compile · 寄存器分块 · QKV 融合 · NS 隔步 | 前三步决定它们值不值得做 |

#### 7.39.4 一条必须坚持的纪律

**抄来的东西不等于对本机有效。** 我们已经吃过这个：CLBlast 在立方形状上能跑 493 GFLOPS，
**但在本机上跑不了它自己的 sample 形状**；NS 的批量版在本机比逐矩阵版慢 11%。

**所以每个"抄"的动作都配一个判据，先写在前面：**

* torch.compile：**编译开销排除后的稳态步时**下降 > 5% ⇒ 留；否则关（编译时间在 CPU 上是分钟级，
  而我们的单步是 3~5 秒，很容易被首步开销骗）
* 寄存器分块：**核显单内核 GEMM ≥ 400 GFLOPS** ⇒ 说明 LDS 不是硬墙，7.28 的结论要改；否则确认
* 优化器融合：短序列（256 token）步时下降 > 10% ⇒ 留（那里优化器占 50%）
* QKV 融合：前向逐位误差 < 1e-5 且整步省时 > 3% ⇒ 留

#### 7.39.5 与既有结论的关系

**不推翻任何东西。** 7.28 的核显结论有六条独立证据（LDS 缺口 · 五条并发全负 · occupancy 无关 ·
小矩阵慢 1.55×），**寄存器分块只检验其中第一条**，即"LDS 是不是硬墙"。**即使它成功，并发那条
（UMA 无独立显存）依然成立**，核显依然不能和 CPU 并用。

**而 CPU 侧的三件（torch.compile · 优化器融合 · QKV 融合）是纯增量**，不依赖核显。

'''

m = re.search(r'^### 7\.39 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.39'
else:
    a = re.search(r'^# 附录', s, re.M)
    if a:
        s = s[:a.start()] + SEC + '\n\n' + s[a.start():]
        how = '插在附录之前'
    else:
        s = s.rstrip() + '\n\n' + SEC
        how = '追加'
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  %s；报告 %d 行' % (how, s.count('\n')))
