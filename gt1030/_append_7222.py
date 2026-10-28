# -*- coding: utf-8 -*-
"""把 §7.222 追加到 CPU_FORGE_TECHNICAL_REPORT.md。"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import io

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

TEXT = r"""

### 7.222 30M AR 图像模型的核显整机移植（M1–M6）与吞吐判决（2026-10-25）

**为什么要做。** §7.221 的结论是「核显那条 `full_gpu_train_hgemm_best.py` 路线只完成速度探针，
不能产模型」：它的模型结构与目标模型 `SmallARImageModelV2` 不同（没有 2D 位置、没有 qk-norm、
没有 ls 残差缩放、层宽也不同），权重根本喂不进 `sample_ar`。本节把真正的 30M AR 模型
（`cond_mode=none / pos_mode=2d / head_mode=flat / use_bos=False / token_order=raster`，30.50M 参数，
grid 32×32 = 1024 token/图）整机搬到 gfx902（6 CU）上，逐层与 torch fp32 对拍，并给出吞吐判决。

**结论先放**：

| 里程碑 | 内容 | 判决 |
|---|---|---|
| M1 | qk-norm / ls 缩放 kernel | ✅ qk_norm rel 4.1e-4、ls 2.8e-4 |
| M2 | 单层 V2 block 前向整链 | ✅ 块贡献 `x2-x` 绝对误差 1.03e-3，cosine 0.992 |
| M3 | 单层反向（qk-norm/ls/融合 SwiGLU） | ✅ dx 6.7e-4、dWqkv 1.9e-2、dWo 1.5e-2、dW12 1.5e-2、dW3 1.5e-2 |
| M4 | 10 层 + head + CE + AdamW 整机 | ✅ **10 步 loss 轨迹最大偏差 4.83e-3**（验收线 0.02），首步 loss 4.910352 vs torch 4.910697（3.5e-4） |
| M5 | 权重导出回 CPU 格式 ckpt + 采样 | ✅ 导出 125 项 rel<5e-3、`load_state_dict(strict=True)` 通过、`sample_ar.py` 直接出图 |
| M6 | 从 CPU ckpt 续训 | ⚠️ 链路通（20 步、513 tok/s、能导出能采样），但**比 CPU 慢 30%** |

**⇒ 功能上，30M AR 模型现在真的能在核显上训练了（这是 §7.221 没做到的）；性能上，它还打不过 CPU，
差距全部集中在 T=1024 的 attention。** 下面的数字都是为了把这句话钉死。

#### 7.222.1 对拍链（这是唯一能证明"真的算对了"的东西）

对拍方法本身踩了一个大坑，值得单独写：**不能拿块的输出 `x2` 去比**。
`x2 = x + ls1·Attn(...) + ls2·SwiGLU(...)`，而 `x` 占 99.9% 的幅度、`ls` 初值只有 1e-2 ——
所以第一版 M2/M3 报的「rel 7.8e-3 通过」是**残差掩盖出来的假象**，块内算成什么样都看不出来。
改成比**块贡献 `x2-x`** 之后立刻暴露 4 个真 bug；而且块贡献只有 1.4e-2 量级、`x2` 又是 fp16 存储
（|x|≈2.5 时 ulp≈1e-3），所以只能用**绝对误差**判定，相对误差没有意义。

| 抓到的 bug | 症状 | 根因 |
|---|---|---|
| `split_q` 里 `int total = T*D*nh*8` | M3 的 dW* 全错（rel≈1.0） | 硬编码 batch=8；我们为 T=1024 用 B=2 ⇒ 线程数远超 kernel 认知的总量，**越界写坏相邻 device buffer** |
| `rms_apply`/`rms_bwd_*` 的 `w` 声明是 `half*` | h1 幅度差 2 倍 | 传了 float32：`1.0f` = `0x3F800000` 被当两个 half 读成 `0.0 / 1.875` 交替 |
| `split_kv` 已经输出 NH 展开后的 K/V，却又套了一层 `expand_kv` | M2 cosine 只有 0.87 | 展开后的索引被当原始索引用，只有前 `B·NKV` 行做过归一化 |
| `dQ = dS @ Kb`、`dK = dSᵀ @ Q` 用了**原始** Q/K | attention 梯度错 45%，dWqkv rel 0.33 | S 是在 qk-norm **之后**的空间里算的，梯度必须用归一化后的 Qn/Kn |
| CE kernel 按 `(M,)` 启动 | 整机 loss 只有 32.1（torch 6.17） | kernel 用 `get_group_id(0)` 当行号 ⇒ `(M,)` 只有 M/256 个工作组，**只算了前 8 行** |
| 梯度整体掉进 fp16 次正规区 | dW12 rel 0.60 | `dlogits≈(p−onehot)/2046≈5e-4`，传到 `dg` 只剩 ~1e-6，而 fp16 最小正规数 6.1e-5 ⇒ 有效位崩掉。修法：CE 梯度整体乘 `CE_SCALE=1024`（AdamW 对梯度尺度不敏感），误差 0.60 → 3.3e-2 |

CE 放大这一条不是"调参"，是**必须做**：不加的话 10 层模型的 dW12/dWqkv 全部 30%~60% 误差，
10 步 loss 轨迹也压不到 5e-3 以内。

#### 7.222.2 驱动缺陷：一个 context 里不能堆很多独立 buffer

这是本轮最费时间的坑，而且**它伪装成"数值错误"**：N≤2 层整机全对，**N≥3 层所有量都变成 0..1023**
（那是我 keep-alive buffer 里的 `np.arange(1024)`）。排查掉的假设：

| 假设 | 否证 |
|---|---|
| 显存不够 | 单块 1.47GB arena + 400 个偏移槽位零错误；2MB×700、4KB×400、混合尺寸×300（1.98GB）全对 |
| 主机内存不够 | avail 6.5GB、commit limit 27.7GB 还剩 13.5GB |
| buffer 数量上限 | 300 个混合尺寸独立 buffer 的纯分配写读全对 |
| 我的某个 kernel 越界 | N=1/N=2 全对；且 `_t_layerwise.py` 显示 **block 0 的输出就已经是垃圾**，与层内计算无关 |
| 多 context 能绕开 | 更糟：3 个 context 起就坏，且新建 context 的 `cl.CommandQueue` 会报 `OUT_OF_HOST_MEMORY`（旧 buffer 释放不干净） |

**实际可用的结论**：本机驱动（gfx902 / Adrenalin）在「一个 context 里放 ~130 个以上大小混杂的
cl_mem」或「开多个 context」时会静默把整条队列搞哑 —— 之后所有 `enqueue_copy` 变 no-op、读回串成
别的 buffer 内容，**且不报任何错误**。而**单 context + 单一大 arena + `get_sub_region(offset,size)`**
是完全稳定的：400 个 sub-buffer / 1.33GB 写读零错误，kernel 直接跑在 sub-buffer 上也对
（`_t_subbuf2/4.py`）。整机因此改成 `ar_v2_multi.py`：一个 3GB arena、每层 ~41 个 sub-buffer、
总共 ~450 个，`CL_DEVICE_MEM_BASE_ADDR_ALIGN=2048` 所以偏移按 4096 对齐。

顺带记两个 API 坑：`mf.COPY_HOST_PTR` 在 buffer 多了以后上传会串数据（一律改「先分配 + 显式 copy
+ 全量回读校验」）；**8 字节回读在这台机器上永远错**，4KB 也会偶发错，只有全量回读可靠。

#### 7.222.3 吞吐判决：4.0s/step vs CPU 3.1s/step

device-only 一步（B=2、T=1024、M=2048 token）：

| 阶段 | 时间 |
|---|---|
| forward | 1.49 s |
| backward | 2.23 s |
| AdamW（29.5M 稠密权重，device） | 0.30 s |
| **合计** | **4.02 s → 509 tok/s** |

CPU（同模型、同 batch，`train_ar_v2.py`）：**3.1 s/step → 569 tok/s**。老的非 V2 iGPU trainer 是 2.15s/step
（但它 attention 小 4 倍、模型少 5M 参数、没有 qk-norm/ls）。逐算子拆解前向（每层每次，`ar_v2_multi_prof.py`）：

| 项 | ms | 有效 GFLOPS |
|---|---|---|
| GEMM W12 (2048×2736×512) | 30.7 | 187 |
| GEMM Wqkv (2048×1024×512) | 16.8 | 128 |
| GEMM W3 (2048×512×1368) | 12.1 | 237 |
| GEMM Wo (2048×512×512) | 4.2 | 258 |
| bGEMM S=QnKnᵀ (16×1024×1024×64) | **39.5** | **54** |
| bGEMM O=PVb (16×1024×64×1024) | **23.9** | **90** |
| softmax_causal (T=1024) | 6.1 | — |
| 其余小 kernel（rms/split/qk-norm/silu/scale-add） | 16.2 | — |
| 层间 x 传输（host 往返 2MB×2） | 3.4 | — |
| **每层合计** | **153** | |

⇒ 前向 1.5s 里 **attention 占 46%**（且 batched GEMM 只有 54~90 GFLOPS，还要物化 S/P 各 32MB），
大 GEMM 占 42%（124~260 GFLOPS）。反向同理，attention 的 batched GEMM 数量翻倍（dP/dV/dQ/dK 四个），
所以反向 2.2s 里 attention 占比更高。

**CLBlast 调参是负结果，必须写下来免得下次再试。** 从 `research/bench_results/clblast_best_configs_1030.json`
出发按我们的真实形状重扫，`KREG>1` 能把名义 GFLOPS 从 124 推到 400~500（Wqkv 472 / W12 454 / W3 457），
**但结果全错**：`_t_override_correct.py` 里 KREG=4/8/16/32 全部 rel≈1.0，只有 KREG=1 时 rel 7.8e-3 正确。
模型上表现为 loss 立刻变成 ln(1024)=6.93。所以那些"加速"是空转 —— **CLBlast 覆盖参数在我们这些
形状上只有 KREG=1 可用，而 KREG=1 的变体最多把 124 提到 135（+9%），没有意义**。
报告 §A.14 里 371~655 GFLOPS / 1.19 TFLOPS 那组数字是 Qwen3-8B 的 M=512、k=4096 形状，
和这里 k=512、n=512~2736 的小形状不是一个问题域。

**优化路线（按性价比）**：

1. **causal-tiled / flash attention**（预期最大）。现在算完整 T×T 再掩码：因果只需要下三角 ⇒ 省一半
   FLOPs；且不需要物化 S/P/dP/dS（每层 4×32MB 的读写）。乐观估计 attention 1.9s/step → 0.6~0.8s。
2. **自写 fp16 tiled GEMM**：报告里自写 half GEMM 能到 352~366 GFLOPS（CLBlast 默认的 2~2.8×）。
3. 小 kernel 合并（现在 16.2ms/层，11%）。

做完 1+2 才谈得上"核显训练 30M 模型比 CPU 快"。在那之前，**生产训练仍留在 CPU**。

#### 7.222.4 工程产物

| 文件 | 用途 |
|---|---|
| `gt1030/ar_v2_ops.py` | M1：qk-norm / ls 缩放 kernel + 对拍 |
| `gt1030/ar_v2_block.py` | M2：单层前向（块贡献对拍） |
| `gt1030/ar_v2_block_bwd.py` | M3：单层反向，13 个中间量逐一对拍 |
| `gt1030/ar_v2_multi.py` | M4：**整机**（单 context + 3GB arena + sub-buffer；10 层 + head + CE + AdamW） |
| `gt1030/ar_v2_multi_selftest.py` | M4 验收：loss / 逐层梯度 / 10 步轨迹 |
| `gt1030/ar_v2_export.py`、`ar_v2_export_test.py` | M5：权重导出 + 可被 torch 严格加载的验证 |
| `gt1030/ar_v2_train.py` | M6：从 CPU ckpt 续训 + 定期导出 |
| `gt1030/ar_v2_multi_bench.py`、`ar_v2_multi_prof.py`、`ar_v2_gemm_sweep*.py` | 吞吐 / 剖面 / 调参（含负结果） |
| `_t_subbuf*.py`、`_t_mixed.py`、`_t_arena2.py`、`_t_crossctx.py` | 驱动缺陷的证据链 |

**Caveat（不要外推）**：M4 的 10 步 loss 轨迹是用 **同一批 2 张图**逐步比对的（等价于过拟合测试），
它验证的是"梯度/更新与 torch 一致"，不是泛化；真训练要等 M6 的吞吐优化后再谈。
另外 ln/qk-norm/ls 这些小向量在 device 侧仍是**冻结**的（它们的梯度需要额外的列归约 kernel），
CPU 那边它们是可训的（实测 ckpt@12000 里 ls 已从 1e-2 长到 0.03~0.07），所以现在的 iGPU 训练
不能直接接管 CPU 的长期训练，必须先补上这 6 类小向量的梯度与 AdamW。
"""

with io.open(P, 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))
