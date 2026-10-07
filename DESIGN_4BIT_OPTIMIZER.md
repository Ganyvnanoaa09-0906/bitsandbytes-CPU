# 4-bit blockwise CPU 优化器 · 设计与开工点

分支：`perf/4bit-blockwise-optimizer`　worktree：`D:\work\bnb-4bitopt`
基准点：`b3f01df`（rmsnorm 两遍 + NT 那次）

---

## 一、目标：**省内存，不是省时间**

先把预期摆正，避免白烧一晚：

| | 数值 | 出处 |
|---|---|---|
| Adam 在训练步里的**时间**占比 | **1~4%** | 报告 §7.12 解析账（30.49M 模型） |
| Adam 每步**搬运** | 732 MB（读 g/m/v，写 m/v/p） | 同上 |
| 优化器状态的**内存**（fp32） | **8 字节/参数**（m + v） | 定义 |
| 8-bit blockwise | ≈ **1.06 字节/参数**（1 + 4/2048） | 现成实现 |
| **4-bit blockwise** | **≈ 0.53 字节/参数** | 本任务 |
| 对 100M 模型 | 800 MB → **53 MB** | — |

**⇒ 价值在"让更大的模型能在这台 15.4 GB 的机器上微调"** ✓
**⇒ 不要在"训练快多少"上抱期望** ✗（除非模型很大 / batch 很小，那时搬运占比才高）

---

## 二、现状（已摸清，别重复摸）

**已有**：
- `csrc/cpu_ops.cpp:3189` `optimizer_update_8bit_blockwise_cpu(...)` ← **本任务的模板**
- `csrc/cpu_ops.h:501` 声明；`csrc/pythonInterface.cpp:917` `coptimizer_update_8bit_blockwise_cpu` 暴露
- `bitsandbytes/optim/` 整个包（`adamw8bit.py` / `optimizer.py` 的 `Optimizer1State`/`Optimizer2State` 机制）
- `csrc/avx2_gemv_4bit.h`（4-bit **推理**的打包/解包约定，**沿用它的约定**）
- 基准：`bench_bnb_opt.py`、`stress_opt.py`；另有本会话做的 `i5build/ab_harness.py`（交替执行 + 参照物归一 + 配对判定，**这台机器分辨力下限约 20~25%**，见报告 §7.15）

**缺**：
- `optimizer_update_4bit_blockwise_cpu` ← **就干这个**
- Python 侧 `optim_bits=4` 的通路

---

## 三、8-bit 模板的结构（照抄这个骨架）

```cpp
void optimizer_update_8bit_blockwise_cpu(
    int optimizer_id, void* g, void* p,
    unsigned char* state1, unsigned char* state2,
    float beta1, float beta2, float beta3, float alpha, float eps,
    int step, float lr, const float* qmap1, const float* qmap2,
    float* absmax1, float* absmax2,
    float weight_decay, float gnorm_scale, bool skip_zeros,
    long long n, int dtype)
{
    // 1) 参数打包进 OptParams P
    //    correction1 = 1 - beta1^step
    //    correction2 = sqrt(1 - beta2^step)
    //    step_size   = -lr * correction2 / correction1     ← 注意符号
    // 2) has_avx2_cpu() 选 AVX2 / scalar 两条路
    // 3) switch (dtype): 0=fp32  1=bf16  2=fp16
    //    → optimizer_8bit_blockwise_avx2<T> / _scalar<T>
}
```

**要点**：`OptParams` 里已经封装了各优化器（adam / adamw / lion / ademamix / lamb / lars / rmsprop / adagrad / sgd）的分支判断，
4-bit 版本应当**复用同一套 `OptParams` 与分支逻辑**，只替换"状态的存取"那一层 ✓

---

## 三之二、★ 读完模板后的重大更正（2026-10-07，以代码为准）

原方案假设"要自己写更新循环" ✗ **错了**。8-bit 模板的真实结构（cpu_ops.cpp:2743~2818）是：

```cpp
// —— 优化器数学：已抽成【共享函数】，4-bit 直接复用，不重写 ——
OptElemResult r = opt_update_element<T>(P, qmap1, qmap2, am1, am2, am3,
                                        c1, c2, c3, g_raw, p_val, one_state);   // :2779
if (r.update_p)
    opt_store<T>(p, i, opt_update_p(P, P.optimizer_id, p_val,
                                    r.s1, r.s2, r.s3, g_wd, one_state));        // :2783

// —— 状态量化：码表【参数化】+ LUT 加速 ——
const unsigned char zc1 = opt_zero_code(qmap1);                                 // :2749
const std::shared_ptr<const LUTEntry> lut1 = get_opt_lut(qmap1, true);          // :2753
state1[i] = opt_sign_fix(qmap1,
                (unsigned char)opt_quant_fast(qmap1, lut1p, s1buf[j]*inv1, true),
                s1buf[j]);                                                      // :2805

// —— 每块的 absmax = 新状态的块内 max|x|（NaN 视为 0）——
n1 = std::fmax(n1, std::isnan(r.s1) ? 0.0f : std::fabs(r.s1));                  // :2788
```

**⇒ 结论：4-bit 版本只需要换【存储与量化层】，优化器数学一行都不用改** ✓✓

要写的只有三件：

| # | 要写的 | 复用 |
|---|---|---|
| 1 | **4-bit 打包存取**：一字节两个 nibble（HIGH = 偶数下标，见 `avx2_gemv_4bit.h:99` 注释 ✓）；`get4(state,i)` / `set4(state,i,code)` | `nibbles_to_lut8` 的解包思路 ✓ |
| 2 | **16 项码表路径**：把 `qmap1/qmap2` 换成 16 项表；16 项时**不需要 LUT**（直接线性或位运算 ✓） | `get_opt_lut` 的结构可参考，但大概率**不需要它** |
| 3 | **入口 + 声明 + 暴露 + Python 通路** | 照 `cpu_ops.cpp:3189` / `cpu_ops.h:501` / `pythonInterface.cpp:917` 抄 |

**⇒ 其余全部复用**：`OptParams`（各优化器分支 ✓）、`opt_update_element` ✓、`opt_update_p` ✓、
`opt_load/opt_store<T>`（fp32/bf16/fp16 三种 ✓）、NaN 处理 ✓、`ademamix` 的三状态 ✓、
OpenMP 分块（注意 2760 那条注释：**parking buffer 必须声明在循环体内** ✗ 否则多线程互相践踏 ✓）

**⚠️ 还有一个必须照抄的细节**：`opt_sign_fix` —— 量化后要按【原始状态值的符号】修正码字（CUDA 同款 ✓）。
4-bit 版本也要做（否则负值会量化到错误的码 ✓）。

**⚠️ 16 项码表的取舍**：8-bit 用 `create_dynamic_map()`（非线性 ✓ 见 :3252 那段教训 ✓）。
4-bit 只有 16 个码 ⇒ **非线性表的收益很小、复杂度代价大** ✗
⇒ **建议纯线性对称量化**（code ∈ 0..15 映射到 [-1,1]，或 ±7 对称 ✓），
   并把这条取舍写进测试注释（判据是"收敛贴合"而非"与 8-bit 一致"✓）

**⇒ 结论：这是"加一个存储层 + 抄三处接口"，不是内核重写** ✓


### 4.1 状态布局

```
state1 (m), state2 (v)  各占 ceil(n/2) 字节：
    一个字节装两个 4-bit 码 —— 低半字节 = 偶数下标，高半字节 = 奇数下标
    约定与 csrc/avx2_gemv_4bit.h 保持一致（不要另立一套）
absmax1, absmax2        每 blocksize=2048 个元素一个 fp32 scale
                        ⇒ 4/2048 = 0.00195 字节/参数
合计 ≈ 0.5 + 0.5 + 0.004 = 1.004 字节/参数（m 与 v 各 0.5）
```

### 4.2 量化

与 8-bit 一致的**块内 absmax 线性量化**，只是位宽从 8 降到 4：

```
code = round(x / absmax * 7)      // 4-bit 有符号，范围 -8..7，或用 0..15 无符号
x'   = code / 7 * absmax
```

**关键决策（开工前要定）**：
- **无符号 0..15 + 块内对称** 还是 **有符号 -8..7**？
  bnb 的 8-bit 用无符号码 + 非对称码表；4-bit 建议 **无符号 0..15 + 块内 (min,max) 双 scale**，
  或 **对称 ±7 + 单 scale**（更简单、更快，代价是精度）。**先用对称 ±7**，简单可验。
- **动态码表 vs 线性**：8-bit 那边踩过坑（见 cpu_ops.cpp:3252 那段注释 —— 硬编码线性表与
  `create_dynamic_map()` 对不上，误差 100% ✗）。**4-bit 用纯线性，并把 scale 显式存盘**，
  不要再引入非线性表。

### 4.3 更新循环（一遍过，这是省搬运的关键）

```
for each block:
    load absmax1[b], absmax2[b]
    for i in block:                       // AVX2 一次 8 或 16 个
        m = unpack4(state1, i) * absmax1 * (1/7)      // 解包 → fp32
        v = unpack4(state2, i) * absmax2 * (1/7)
        g = load_grad(i) * gnorm_scale                 // 支持 fp32/bf16/fp16
        // ---- 与 8-bit 完全相同的更新式，只是 m/v 来自解包 ----
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g*g
        p -= step_size * m / (sqrt(v) + eps)  (+ weight_decay)
        // ---- 重新量化并【原地写回】 ----
        state1[i] = pack4(m / absmax1_new)
        state2[i] = pack4(v / absmax2_new)
    update absmax1[b], absmax2[b]         // 用块内新的 max|m| / max|v|
```

**注意**：absmax 是"边更新边用"还是"整块先算后写"要定清楚 ——
8-bit 那边的做法**照抄**（避免自创 ✓）。

---

## 五、验证与测量（按顺序，一步都不能跳）

1. **数值正确性**
   - 先写**单块 unittest**（n=4096、blocksize=2048、已知输入 ✓）对照 numpy 参考实现
   - 判据：**不是逐位一致**（量化本来就不一致 ✗），而是
     **量化误差的上界**（|x - dequant(quant(x))| ≤ absmax/14 对 ±7 对称量化 ✓）
   - 再三方对照：fp32 AdamW vs 8-bit vs 4-bit 跑 **200 步**同一模型
     ⇒ 判据 = **loss 曲线贴合**（不是相等 ✓）
2. **内存**
   - 直接量：`state1.nbytes + state2.nbytes + absmax.nbytes` ✓ 与 fp32 对比 ✓
   - 这是本任务的**主指标** ✓
3. **时间**
   - 用 `i5build/ab_harness.py`（**交替执行 + 参照物归一 + 配对判定** ✓）
   - ⚠️ 这台机器的分辨力下限 ~20~25%（报告 §7.15）⇒ **小差异一律记为不可判定** ✓
4. **端到端**
   - 拿 `train_ar_v2.py` 或任一真实训练，把优化器换成 4-bit，跑 500 步
   - 看 loss 曲线与 fp32 是否贴合 ✓ + 内存峰值下降多少 ✓

---

## 六、开工点（精确坐标，已核实 2026-10-07）

```
cpu_ops.cpp:2612   struct OptParams                    ← 复用（含各优化器分支判断）
cpu_ops.cpp:2743   optimizer_8bit_blockwise_scalar<T>   ← 模板 A（先读这个，短，逻辑清楚）
cpu_ops.cpp:2914   optimizer_8bit_blockwise_avx2<T>     ← 模板 B（AVX2 版，照抄结构）
cpu_ops.cpp:3189   optimizer_update_8bit_blockwise_cpu  ← 对外入口（4-bit 版照它的样子写）
cpu_ops.cpp:3252   8-bit 码表硬编码的教训注释           ← 必读，别再踩
cpu_ops.cpp:3264   ARM64 漏守卫的教训注释               ← 4-bit 辅助函数必须加 #if 守卫
cpu_ops.h:501      声明处
pythonInterface.cpp:917  coptimizer_update_8bit_blockwise_cpu  ← 暴露写法照抄
```

★★ **4-bit 打包已经有现成实现，沿用它的约定，别自创** ★★

```
csrc/avx2_gemv_4bit.h:85    lut_planes(lut, p0, p1, ...)
csrc/avx2_gemv_4bit.h:99    注释明确写出约定：
                            "Expand 4 packed bytes (8 nibbles,
                             HIGH nibble = even element) into 8 LUT floats"
csrc/avx2_gemv_4bit.h:101   nibbles_to_lut8(p4, p0, p1, ...)   ← 解包辅助，直接用
                            （内部：mask4=0x0F；_mm_unpacklo_epi8(hi, lo) 得到 8 个索引）
```

⇒ **打包约定：一字节两个 nibble，HIGH = 偶数下标元素** ✓ 我方案里猜的"低半字节=偶数"是**反的** ✗
   **以 avx2_gemv_4bit.h 的注释为准** ✓✓

⇒ **解包路径已经写好** ✓ 4-bit 优化器缺的主要是 **"更新后重新量化并原地写回"** 那一半 ✓
   （推理只需解包 ✓ 优化器还需打包 ✓ —— 打包方向可能是本次要新写的唯一 ALU 部分 ✓）

```
1. 读 cpu_ops.cpp:2743（scalar 模板）→ 2743~2913 是完整逻辑
2. 读 avx2_gemv_4bit.h:80~130（打包/解包辅助 + 约定）
3. 新建 optimizer_4bit_blockwise_{scalar,avx2}<T>（可放新头文件，跟随现有文件划分）
4. 写 optimizer_update_4bit_blockwise_cpu 入口（照 :3189 的样子）
5. 声明进 cpu_ops.h，暴露进 pythonInterface.cpp（照 :917）
6. Python：optim/optimizer.py 的 optim_bits 分支 + 一个 4-bit state 类
7. 测试：tests/test_optimizer_4bit_cpu.py（照 tests/test_rmsnorm_fwd_cpu.py 的风格）
```

**编译**：跟 fork 现有的手工构建流程（`build_manual/` 或 `python setup.py build_ext`）
**注意**：MSVC 的 OpenMP 2.0 限制（循环变量必须在 pragma 外声明 ✓）；
       `.bat` 必须纯 ASCII ✓；MSVC 要 `/utf-8` ✓（这几条本会话都踩过 ✓）

---

## 七、已知的坑（本会话总结，直接抄，别再踩）

| 坑 | 教训 |
|---|---|
| 8-bit 码表硬编码 | cpu_ops.cpp:3252 那段：融合核与量化器对不上 ⇒ **码表当参数传，别硬编码** |
| 定规矩前先量 | 本机分辨力下限 ~20~25% ⇒ **小于它的优化无法验证**（报告 §7.15） |
| 逐位一致 vs 收敛一致 | 量化优化器**不可能逐位一致** ⇒ 判据要写"loss 曲线贴合" ✓ |
| ARM64 构建 | 4-bit 辅助函数**必须加 `#if defined(__AVX2__)` 守卫**（cpu_ops.cpp:3264 那段就是漏守卫的教训） |
| 提权/引号/编码 | 见 `D:\work\HANDOVER_2026-10-05.md` 第六节（20 条） |
