# -*- coding: utf-8 -*-
"""追加 §7.222.21：真 flash attention 实测 —— 在我们这个形状上它比现有三 kernel 路径慢 1.7 倍。"""
import io

TEXT = r"""

#### 7.222.21 真 flash attention 实测：**22.4 ms vs 现有 12.7 ms —— flash 输了 1.7 倍**

按计划把 fwd 的 S/softmax/O 融成一个 kernel（不落盘 S、P）。实现细节：
一个工作组 = 64 行 × 64 头维，16×16 线程、每线程 4×4 的 O 累加器；
local 里 Qt(8KB) + KV(8KB，先放 K 算完 S 再换 V) + St(8KB，先 S 后 P) + 行 max/和 ≈ 28.75KB；
每个列块 64 做在线 softmax（行 max 归约 → P → 行和归约 → O 累加），末尾除 lrow。

**踩到的三个真 bug**（都靠和现有路径逐值对比才发现）：
1. `acc[4][4]` 被同时当成 S 的临时累加器和 O 的持久累加器 → 必须拆成 `sacc` / `oacc`；
2. 只在"整块"层面按列块截断，**对角块内部 c > r 没掩码** ⇒ rel=0.99，补上掩码后 rel=6.5e-4；
3. launch 又是 `global=(256, BH)` 而 `local=(16,16)` ⇒ dim1 只有 1 个工作组，只算了第 0 个 (batch,head)
   （这是本机这个 API 上第三次踩同一个坑：**global 必须是 组数 × 组大小**）。

**结果（同一进程、同一时刻对拍）**：

| 路径 | 时间/层 | 数值 |
|---|---|---|
| 现有三 kernel（自写 `sgemm_nt64` + `softmax_causal_v` + `sgemm_nn_bt`，都做了因果裁剪） | **12.68 ms** | 基准 |
| 真 flash（单 kernel，融合） | 21.64 ms | rel 6.5e-4 ✓ |

⇒ **flash 慢 0.59×（即慢了 71%）**。和 CLBlast 版三 kernel（33.3 ms）比 flash 是 1.48×，
但**我们已经不跑那个版本了** —— 拿 flash 去比"当前实现"就输了。

**为什么在这台机器/这个形状上 flash 不划算**：

1. **形状太"薄"**：T=1024、k=HD=64，attention 的 FLOPs 只有 2.15 GFLOP/层（前向）。
   现有路径已经把因果裁剪用上（S/O 各只算下三角）；flash 能省的只是 S/P 那 ~64 MB 的落盘流量，
   按实测带宽 ~19 GB/s 值 3.4 ms/层 —— 而为了省这 3.4 ms，flash 必须在 kernel 内部用
   **标量 local 读**（S/P 由 WG 自己产生，没法像独立 GEMM 那样预转置成 vload4 友好布局），
   每 kk 要 8 条标量 LDS 换 16 次 FMA（现有 kernel 是 2 条 vload4 换 16 次）。
2. **occupancy 被 local 内存压死**：28.75 KB/WG ⇒ 每 CU 只能驻 1 个工作组（32KB 上限），
   而现有三个 kernel 每个只要 4~16 KB、能驻 2~3 个。
3. GCN 上"少搬数据"换"多算指令"这个交易不划算 —— 这正是 §7.222.18 里小 kernel 向量化能赢、
   而 flash 会输的同一个原因（赢的是**指令效率**，不是带宽）。

**结论**：attention 这块在我们这个形状上已经基本到顶（前向 12.7 ms、反向约 17.7 ms，合计 ~300 ms/步
= 25%）。想再往下不是"更聪明的算法"，而是"让 GEMM 本身跑得更快"，而后者卡在 CLBlast 已到本机实际可达值
（§7.222.20 的双缓冲尝试也因 OpenCL 的 1-D strided copy 语义而无法落地）。

⇒ **1.216 s/步（1030 分数 0.485）可以当作这条技术路线的实际下限附近**：
其中 46% 是大 GEMM（CLBlast 上限）、25% 是 attention（已比朴素实现快 2.6×）、29% 是其余开销。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace('| 负结果 | 向量化 softmax 无效（带宽瓶颈 19 GB/s）；GEMM 双缓冲被 `async_work_group_strided_copy` 的 1-D 语义卡住 |',
              '| 负结果 | 向量化 softmax 无效（带宽瓶颈 19 GB/s）；GEMM 双缓冲被 `async_work_group_strided_copy` 的 1-D 语义卡住；'
              '**真 flash attention 比现有三 kernel 慢 1.7×（22.4 vs 12.7 ms/层）** |')
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
