# 融合 elementwise 内核族（CPU，AVX2+FMA）

> 本文描述的是 **v0.50.2.dev0 之后新增**的一批 CPU 融合内核。
> 它们**不在**上游 bitsandbytes 里，是本 fork 为「纯 CPU 训练/推理」加的。
> 所有收益数字都来自本仓的实测脚本（可复跑），不是估算。

---

## 0. 为什么会有这一族内核

实测（`probe_step_breakdown.py`，R5-4500U，纯 CPU 训练一个 30M 级模型）：

```
训练步 2.819 s 的构成:
    GEMM        63.6%     ← 已经达同形状纯 GEMM 基准的 92%，没有空间
    elementwise 20.8%     ← ★ 唯一还有空间的一块
    attention    7.6%
    其它         ~8%
```

而 eager 模式下 `RMSNorm` 会展开成 **6 个 aten 算子**
（`pow → mean → add → rsqrt → mul → mul`），每个算子都要**完整读写一遍激活张量**。
激活张量本身不大（`B×T×D×4` 字节），所以这是**纯带宽浪费**：
把 6 趟压成 1 趟，就是这一族内核的全部意义。

**边界要说清**：它优化的是**带宽**，不是算力。
所以对「算术强度已经远高于拐点」的负载（例如卷积为主的 UNet）**没有收益**——
那种负载该用别的办法（见 `TECH_REPORT.md` §3 的卷积反向一节）。

---

## 1. Python 公开接口：`fused_cpu.py`

```python
import fused_cpu
fused_cpu.available()          # DLL 里有没有 cfused_* 导出（没编译就 False）
```

| 函数 | 语义 | 对应 eager 写法 |
|---|---|---|
| `fused_rmsnorm(x, weight=None, res=None, eps=1e-6)` | `(x + res) * rsqrt(mean((x+res)²) + eps) * weight` | 6 个 aten 算子 |
| `fused_swiglu(gate, up)` | `silu(gate) * up` | `F.silu(gate) * up` |
| `fused_add_scale(x, y, scale=None)` | `x + scale * y`（`scale=None` 时退化为 `x + y`） | `x + scale * y` |
| `fused_add_scale_rmsnorm(x, y, scale, weight, eps=1e-6)` | `RMSNorm(x + scale*y)` 一步出 | 残差 + norm 两趟 |

★ **四个都是 `torch.autograd.Function`，前向反向都有**——可以直接放进训练图，
不需要手写 backward。

**典型用法（Transformer block）**：

```python
import fused_cpu

# 残差 + LayerScale + RMSNorm 一步完成（原来是两趟读写）
h = fused_cpu.fused_add_scale_rmsnorm(x, attn_out, self.ls1, self.ln1_w, eps=1e-6)

# SwiGLU MLP
gate = self.w1(h); up = self.w3(h)
h2 = self.w2(fused_cpu.fused_swiglu(gate, up))

x = fused_cpu.fused_add_scale(h, h2, self.ls2)      # x + ls2 * h2
```

**数值口径（可核对）**：融合核对拍 eager 的 `max|Δ|` 在 **1e-7 ~ 1e-6** 量级
（fp32 累加顺序不同所致），`state_dict` 键名不变 ⇒ checkpoint 可安全互载。
自检脚本：`py -3.11 test_fused_kernels.py`（对每个核对拍前向+反向并计时）。

---

## 2. 一键接线：`enable_fused.py`（推荐）

不想手改模型代码时用这个——它把融合核 **monkey-patch** 到模型类的算子链上：

```python
import enable_fused
enable_fused.enable()             # 只融合 RMSNorm（改动最小、最稳）
enable_fused.enable(block=True)   # 再加残差 + LayerScale
# enable_fused.enable(rmsnorm=True, swiglu=True, block=True)   # 全开
```

**实测收益**（交替测量 6 轮，取配对比值的中位数；`enable_fused.py` 头部记录）：

```
只融合 RMSNorm                 +5.0%（保守） ~ +8.0%（中位）
RMSNorm + SwiGLU + 残差        +5.5%（保守） ~ +7.1%（中位）
⇒ 两者接近 ⇒ 默认【只开 RMSNorm】（改动面最小）
```

⚠️ **两条使用约束**：
1. **必须在创建模型实例之前调用** —— 它 patch 的是**类方法**（对已建实例也生效，
   但顺序清晰些更好排查）
2. 依赖 `fused_cpu.available()`；DLL 里没有 `cfused_*` 导出时会直接
   `RuntimeError('DLL 里没有融合核 —— 需要重新编译 bnb（含 cfused_* 导出）')`
   ⇒ 先按 `QUICKSTART.md §2` 重新编译

---

## 3. C 侧导出（给写 C/C++ 或别的语言绑定的人）

`csrc/pythonInterface.cpp` 里的导出包装（`c` 前缀）：

```c
long long cfused_rmsnorm_fwd_cpu(const float* x, const float* res, const float* weight,
                                 float* out, float* xs, long long M, long long D, float eps);
long long cfused_rmsnorm_bwd_cpu(const float* x, const float* weight, const float* dout,
                                 const float* xs, float* dx, float* dw,
                                 long long M, long long D);
long long cfused_swiglu_fwd_cpu(const float* gate, const float* up, float* out, long long n);
long long cfused_swiglu_bwd_cpu(const float* gate, const float* up, const float* dout,
                                float* dgate, float* dup, long long n);
long long cfused_add_scale_cpu(const float* x, const float* y, const float* scale,
                               float* out, long long M, long long D);
long long cfused_add_scale_inplace_cpu(const float* x, const float* y, const float* scale,
                                       long long M, long long D);          /* 原地，省一趟 */
long long cfused_add_scale_rmsnorm_cpu(const float* x, const float* y, const float* scale,
                                       const float* weight, float* out, float* xs,
                                       long long M, long long D, float eps);
/* strided 版：gate/up/out 的行跨步与列跨步可不同（吃 chunk 视图，零拷贝） */
long long cfused_swiglu_strided_fwd_cpu(const float* gate, long long gs0, long long gs1,
                                        const float* up,  long long us0, long long us1,
                                        float* out, long long os0, long long M, long long D);
long long cfused_swiglu_strided_bwd_cpu(...);
```

**参数约定**（容易踩的两个）：
- `M` 是**行数**（`B*T`），`D` 是**最后一维**；要求 `x` 行主序、`D` 连续
- `xs` 是 RMSNorm 前向**必须**输出的中间量（`rsqrt(mean(x²)+eps)`，长度 `M`），
  反向要用它 —— 不能省，也不要用 `torch` 的 `save_for_backward` 重复存

**strided 版的意义**：`F.silu(a) * b` 里的 `a`、`b` 常常是同一个大张量切出来的
chunk（QKV 融合后的视图）。strided 版直接吃这个视图，**不复制**。

---

## 4. NT store（非临时存储）：一个被自动应用的带宽优化

这批内核里凡是**大数组纯写**的地方（8-bit 反量化写出、`fused_add_scale`、
`fused_swiglu`），都会在满足条件时走 `_mm256_stream_ps`（NT store）。

**原理**：普通存储会触发 write-allocate（RFO，先把目标 cache line 读进来再写回），
凭空多出 1/3 的内存流量。NT store 绕过 cache，目标行不读直接写。

**实测（`membw3.c`，纯 AVX2，48/192 MB 足迹，6 线程）**：

| kernel | 普通存储 | NT store | 比值 |
|---|---|---|---|
| copy | 13.40 GB/s | **25.40** | 1.90× |
| triad | 16.09 | **23.86** | 1.48× |

**但阈值是运行时发现的，不要硬编码**：

```c
use_nt = bnb_is_aligned_for_nt(out)                        /* 32 字节对齐 */
      && (n * sizeof(T) >= bnb_nt_threshold_bytes());      /* 输出够大 */
```

```
· 输出 <= 1.0 MB : 普通存储赢（NT 亏 0.69~0.73×）—— 因为它把还要用的数据踢出 cache
· 输出 >= 4.2 MB : NT 赢 2.0~3.1×
· 阈值按**运行时 L3 大小**推导（L3 在 8/12/32+ MB 之间变）
```

⚠️ **反面教材（同一天实测）**：优化器的 `p` 写入**不能**用 NT —— 那里是读-改-写，
cache line 已经被读过，普通 store 只是标脏，NT 反而强制刷出 ⇒ **倒退 21%**。

---

## 5. `gemv_fp16w_inference_cpu_*`：fp16 权重 GEMV/GEMM（**已导出，尚未接 Python**）

```c
void gemv_fp16w_inference_cpu_fp32(const void* A, const void* W, void* out,
                                   long long M, long long N, long long K,
                                   long long lda, long long ldb, long long ldc);
/* 另有 _bf16 / _fp16 两个输出精度版本 */
```

**用途**：权重存 fp16、算子在 fp32 里累加 —— 权重流量减半，而 fp32 的 FMA 吞吐不变。
这正是 §4/`TECH_REPORT.md` 里「4-bit 推理在 AVX2-only CPU 上净亏」那个结论的**替代解**：
4-bit 需要解包指令（AVX2 上没有 VNNI，解包反而成为瓶颈），fp16 不需要。

**⚠️ 当前状态要说清**：
```
· C 侧：已实现、已导出（pythonInterface.cpp L886-904）、已编进 DLL
        三个变体分别是 gemv_fp16w_inference_cpu_fp32 / gemv_fp16w_inference_cpu_bf16
        / gemv_fp16w_inference_cpu_fp16（按输出精度分）
· Python 侧：**没有调用点**（全仓 grep `gemv_fp16w` 只命中 C 侧）
⇒ 目前只能从 C/C++ 或 ctypes 直接调用；还没有 `LinearFP16W` 之类的 nn.Module 封装。
   若你要在 Python 里用，最小做法是照 fused_cpu.py 的样子写个 ctypes 绑定。
```
★ 记录这一条的目的：**不要让读者以为它已经能用** —— 已导出 ≠ 已接线。

---

## 6. 本族的已知边界

| 边界 | 说明 |
|---|---|
| 只省**带宽**，不省算力 | 对算术强度远高于拐点的负载（卷积为主）收益接近 0 |
| 需要**重新编译** | DLL 里没有 `cfused_*` 时 `fused_cpu.available()` 返回 False |
| `enable_fused` 要在建模型前调用 | patch 的是类方法 |
| NT store 有**阈值与对齐**要求 | 见 §4；小输出会倒退 |
| `gemv_fp16w` 未接 Python | 见 §5 |
| 数值是 fp32 舍入级差异 | `max|Δ| ~1e-7`，checkpoint 可互载 |

---

## 7. 复跑这些数字

```bat
py -3.11 test_fused_kernels.py        :: 四个核的对拍（前向+反向）+ 计时
py -3.11 bench_fused_final.py         :: 端到端（RMSNorm + 残差都走融合）
py -3.11 bench_fused_e2e3.py          :: v1/v2/v3 三种接法对比
py -3.11 bench_l3_vs_swiglu.py        :: 融合 vs 单独算（隔离对照）
py -3.11 bench_r5_threads.py          :: 线程数（本机 5 线程最优，6 反而慢 5.9%）
```
