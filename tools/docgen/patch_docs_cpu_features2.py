"""Second docs_cpu pass: AVX-512 backend + training integration for 2/3-bit + L3 mechanism table.

The first pass added 4.11-4.13 / 3.9-3.11 / 3.8-3.10 (iGPU CLBlast, AdamW4bit, 2/3-bit). This pass
adds the remaining workspace workstream that had no user-facing docs at all -- the AVX-512 build --
and folds in two measured tables that the earlier root-docs revision had (L3-sharing mechanism, and the
"when does the device win" applicability interval).
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DOCS = r'D:\work\bitsandbytes-CPU\bitsandbytes\docs_cpu'

QS_ZH = '''### 4.14 AVX-512 构建（实验性）

CPU 后端在 AVX2 之外还有一套手写 AVX-512 内核（4-bit 反量化的 16-lane 路径、bf16 GEMV 的
tinygemm 路径），默认关闭。构建方式（要求构建机自己能执行 AVX-512 探针）：

```bat
:: CMake 构建（Linux/GCC），BNB_ENABLE_AVX512 默认 OFF
cmake -DBNB_ENABLE_AVX512=ON ..
```

**已验证的事实（远程 AVX-512 机器实测）**：

| 项 | 结果 |
|---|---|
| 数值正确性 | 5 个内核与 AVX2 构建**逐位一致**（误差 0.00e+00） |
| 分派 | 库内 `has_avx512f_cpu()=True`，受保护内核的守卫成立 |
| 硬件峰值 | 全宽 512 位机器：fp32 FMA **2.02×** AVX2、BF16 点积 **2.71×**（同编译器、单线程） |
| 真实内核 | 现有 memory-bound 内核 **0.97~1.02×**（见 `TECH_REPORT.md` §3.11） |

⚠️ **注意**：`-mavx512f` 是**整编译单元**级开关，会让普通内核（dequantize、优化器、rmsnorm）
被编译器自动向量化成 zmm → 在 AVX2 机器上**一调用就 SIGILL**。所以 AVX-512 二进制不能发给
AVX2 机器，必须按 CPUID 选择加载（本 fork 尚未把"双库+选路"接入安装包，属于待办）。

'''

QS_EN = '''### 4.14 AVX-512 builds (experimental)

Beyond AVX2 the CPU backend has a second set of hand-written AVX-512 kernels (the 16-lane 4-bit
dequantise path and the bf16 GEMV tinygemm path), off by default. Build them with the opt-in switch
(the build host must be able to *run* an AVX-512 probe):

```bat
:: CMake build (Linux/GCC); BNB_ENABLE_AVX512 defaults to OFF
cmake -DBNB_ENABLE_AVX512=ON ..
```

**Verified facts (measured on a remote AVX-512 machine)**:

| Item | Result |
|---|---|
| Correctness | five kernels **bit-identical** to the AVX2 build (max diff 0.00e+00) |
| Dispatch | the library reports `has_avx512f_cpu()=True`, so the guarded kernels' guard holds |
| Hardware ceiling | on a full-width 512-bit CPU: fp32 FMA **2.02x** AVX2, BF16 dot product **2.71x** (same compiler, single thread) |
| Real kernels | **0.97-1.02x** on today's memory-bound kernels (see `TECH_REPORT_EN.md` 3.11) |

⚠️ **Warning**: `-mavx512f` is a **translation-unit-wide** switch, so ordinary kernels
(dequantise, optimizer, rmsnorm) get auto-vectorised to zmm and **SIGILL on an AVX2-only CPU** the
moment they are called. An AVX-512 binary therefore cannot be shipped to AVX2 machines; it has to be
loaded through a CPUID check (the dual-library selector is not wired into the installer yet -- it is
on the todo list).

'''

GUIDE_ZH = '''### 3.12 AVX-512 后端：分派、守卫与"为什么需要构建机自己支持"

**代码形态**：手写 AVX-512 内核包在 `#if defined(__AVX512F__)` 里，入口再套一层运行时守卫
`if (has_avx512f())`；基础路径则用 `#pragma GCC target("avx2,fma,no-avx512f")` 防止编译器在
普通函数里发射 EVEX 编码。`CMakeLists.txt` 的 `BNB_ENABLE_AVX512` 默认 OFF。

**为什么必须"构建机自己跑得动 AVX-512"**：`-mavx512f` 是**整 TU** 级编译开关。实测在本机
（AVX2-only）强行编译：产物 330 KB、1861 条 zmm 指令，一跑自检就 `Illegal instruction (132)`；
把 zmm 归属到函数后发现，崩溃点不是手写内核，而是 `dequantizeBlockwise4bit/8bitCpu`、
`fused_rmsnorm_bwd_cpu`、`optimizer_update_8bit_blockwise_cpu` 这些**每台机器都会调用**的普通
函数被自动向量化了。⇒ 这就是 CMake 要求"探针在构建机上真跑一遍"的物理原因。

**数值与分派已验证**：在真正的 AVX-512 机器上，5 个内核与 AVX2 构建逐位一致；AVX-512 库自己的
`has_avx512f_cpu()` 返回 True（即守卫成立）。

**收益边界**：硬件峰值是真的（全宽 512 位机器 fp32 2.02×、BF16 点积 2.71×），但现有 AVX-512
内核都是 memory-bound 的（反量化、优化器状态更新、4-bit GEMV），实测只有 0.97~1.02×。
**要拿到峰值收益，必须新写 compute-bound 的 AVX-512 内核（BF16 / VNNI / GEMM），而不是把现有
内核用 AVX-512 编译一遍。**

'''

GUIDE_EN = '''### 3.12 The AVX-512 backend: dispatch, guards, and why the build host must support it

**Code shape**: the hand-written AVX-512 kernels sit inside `#if defined(__AVX512F__)` and their entry
points are additionally guarded at run time by `if (has_avx512f())`; baseline functions are wrapped in
`#pragma GCC target("avx2,fma,no-avx512f")` so the compiler cannot emit EVEX-encoded instructions in
code that every CPU calls. `BNB_ENABLE_AVX512` in `CMakeLists.txt` defaults to OFF.

**Why the build host itself must be able to run AVX-512**: `-mavx512f` is a **per-translation-unit**
switch. Compiling it on the AVX2-only development machine produces a 330 KB library with 1861 zmm
instruction lines, and the self-test dies with `Illegal instruction (132)`; attributing the zmm lines
to functions shows the crash sites are not the hand-written kernels but the ordinary functions every
machine calls -- `dequantizeBlockwise4bit/8bitCpu`, `fused_rmsnorm_bwd_cpu`,
`optimizer_update_8bit_blockwise_cpu` -- all auto-vectorised. That is the physical reason the CMake
option requires a probe that actually runs on the build host.

**Numerics and dispatch are verified**: on a real AVX-512 machine five kernels are bit-identical to the
AVX2 build, and the AVX-512 library's own `has_avx512f_cpu()` returns True (the guard holds).

**Where the gain stops**: the hardware ceiling is real (on a full-width 512-bit CPU, fp32 2.02x and
BF16 dot product 2.71x), but today's AVX-512 kernels are all memory-bound (dequantise, optimizer state
update, 4-bit GEMV), so the measured kernel speedup is only 0.97-1.02x. **Capturing the peak requires
new compute-bound AVX-512 kernels (BF16 / VNNI / GEMM), not recompiling the existing ones with AVX-512
flags.**

'''

REPORT_ZH = '''### 3.11 AVX-512 后端实测：数值全对，现有内核 0.97~1.02×，峰值要等新内核

**第一次真机验证**（Ubuntu 22.04 / gcc 11.4 / 全宽 512 位 CPU / 2 核容器）：

| 项 | 结果 |
|---|---|
| 构建 | AVX2 基线 256 KB、AVX-512 变体 284 KB（同一份源码，只差 flags） |
| 分派 | `has_avx512f_cpu()=True`、`has_avx512bf16_cpu()=True` |
| 正确性 | quantize8 / dequantize8 / gemm_8bit / gemm_8bit_M64 / gemv_4bit / adamw8bit **6/6 逐位一致** |
| 硬件峰值（单线程） | fp32 FMA 113.7 → **229.3 GFLOPS（2.02×）**；BF16 点积 **308.6 GFLOPS（2.71× AVX2-fp32）** |
| 真实内核 | quantize8 0.99× · dequantize8 0.99× · gemm_8bit 0.97× · gemm_M64 1.02× · gemv_4bit 1.01× · adamw8bit 1.02× |

**为什么峰值 2× 而内核只有 1.0×**：这些内核全是**搬字节**的（反量化、优化器状态、M=1 的 GEMV），
没有一处吃 FMA 峰值；连特意改成的算力型 `gemm_8bit_M64` 也只有 1.02×，因为现有实现是"逐块
反量化 B 再 GEMM"，时间花在 LUT/gather 与 fp32 中间缓冲上。**⇒ AVX-512 的收益不在"重编译"，
而在"为 compute-bound 路径新写内核"（BF16 点积 / VNNI int8）。**

**另一个必须记住的坑**：在本机（AVX2-only）强行编译 AVX-512 版，产物 1861 条 zmm 指令，一跑就
SIGILL；zmm 归属证明崩溃点在**普通函数**（dequantize/rmsnorm/optimizer）而不是手写内核——
`-mavx512f` 是整 TU 开关，这是"构建机必须自己支持 AVX-512"的物理原因。

'''

REPORT_EN = '''### 3.11 The AVX-512 backend measured: numerics correct, current kernels 0.97-1.02x, peak waits for new kernels

**First real-machine validation** (Ubuntu 22.04 / gcc 11.4 / full-width 512-bit CPU / 2-core container):

| Item | Result |
|---|---|
| Build | AVX2 baseline 256 KB, AVX-512 variant 284 KB (same sources, flags only) |
| Dispatch | `has_avx512f_cpu()=True`, `has_avx512bf16_cpu()=True` |
| Correctness | quantize8 / dequantize8 / gemm_8bit / gemm_8bit_M64 / gemv_4bit / adamw8bit **6/6 bit-identical** |
| Hardware ceiling (single thread) | fp32 FMA 113.7 -> **229.3 GFLOPS (2.02x)**; BF16 dot product **308.6 GFLOPS (2.71x AVX2-fp32)** |
| Real kernels | quantize8 0.99x, dequantize8 0.99x, gemm_8bit 0.97x, gemm_M64 1.02x, gemv_4bit 1.01x, adamw8bit 1.02x |

**Why the ceiling is 2x while the kernels stay at 1.0x**: all of these kernels move bytes
(dequantise, optimizer state, M=1 GEMV) and none reaches the FMA ceiling; even the deliberately
compute-heavy `gemm_8bit_M64` only reaches 1.02x, because the current implementation dequantises B block
by block and then runs a GEMM, so the time goes to LUT/gather and the fp32 scratch buffer.
**=> the AVX-512 payoff is not in "recompiling" but in "writing new compute-bound kernels"
(BF16 dot product / VNNI int8).**

**One more trap worth remembering**: forcing the AVX-512 build on the AVX2-only box yields 1861 zmm
instruction lines and SIGILLs immediately; attributing the zmm lines shows the crash sites are the
**ordinary** functions (dequantise/rmsnorm/optimizer), not the hand-written kernels -- `-mavx512f` is a
whole-TU switch, and that is the physical reason the build host itself must support AVX-512.

'''

L3_ZH_INSERT = '''**为什么不能用"CPU+核显并发"**：两者共享 8 MB L3，CPU 工作集把核显的工作集挤出去。
三档对照（CPU 单线程负载 → 核显相对空闲时的吞吐）：纯 ALU 自旋（工作集 0）**1.03×**、
`torch.mm` 256³（0.8 MB）**1.96×**、`torch.mm` 2048³（48 MB）**6.08×**。
⇒ 一个满功耗零访存的负载只让核显掉 3%，而工作集越大掉得越多 —— **机制是共享 L3 被挤，不是功耗仲裁**
（功耗仲裁是开关式的，缓存争抢是连续可调的）。

'''

L3_EN_INSERT = '''**Why CPU+iGPU concurrency does not work**: the two share the 8 MB L3, and the CPU working set
evicts the iGPU's. Three CPU loads (single-thread) against the idle-iGPU throughput: pure ALU spin
(working set 0) **1.03x**, `torch.mm` 256^3 (0.8 MB) **1.96x**, `torch.mm` 2048^3 (48 MB) **6.08x**.
=> a full-power, zero-memory load only costs the iGPU 3%, while a larger working set costs more --
**the mechanism is L3 sharing, not power arbitration** (power arbitration is a switch, cache
contention is a continuous knob).

'''

EDITS = [
    ('QUICKSTART.md', '## 5. 硬盘均衡负载（disk_balancer，SSD 保护）', QS_ZH + '## 5. 硬盘均衡负载（disk_balancer，SSD 保护）'),
    ('QUICKSTART_EN.md', '## 5. Disk load balancer (disk_balancer, SSD protection)', QS_EN + '## 5. Disk load balancer (disk_balancer, SSD protection)'),
    ('TECHNICAL_GUIDE.md', '## 4. 内存策略：为什么「量化冻结层」而不是「降权重存储精度」', GUIDE_ZH + '## 4. 内存策略：为什么「量化冻结层」而不是「降权重存储精度」'),
    ('TECHNICAL_GUIDE_EN.md', '## 4. Memory strategy: why "quantized frozen layers" not "lower-precision weight storage"', GUIDE_EN + '## 4. Memory strategy: why "quantized frozen layers" not "lower-precision weight storage"'),
    ('TECH_REPORT.md', '## 4. 生图训练实测（R5-4500U）', REPORT_ZH + '## 4. 生图训练实测（R5-4500U）'),
    ('TECH_REPORT_EN.md', '## 4. Image-generation training measured (R5-4500U)', REPORT_EN + '## 4. Image-generation training measured (R5-4500U)'),
    # L3 mechanism paragraph, replacing the shorter overlap sentence in the iGPU sections
    ('TECHNICAL_GUIDE.md',
     '**为什么不能用"CPU+核显并发"**：两者共享 8 MB L3，CPU 工作集把核显的工作集挤出去 ⇒\n'
     'iGPU 只剩 40~54%、CPU 剩 53~61%，并发 1.1~1.3× 不如顺序执行 1.51×。\n',
     L3_ZH_INSERT),
    ('TECHNICAL_GUIDE_EN.md',
     '**Why CPU+iGPU concurrency does not work**: the two share the 8 MB L3, and the CPU working set\n'
     'evicts the iGPU\'s, so the iGPU retains 40-54% and the CPU 53-61%; concurrency gives 1.1-1.3x\n'
     'against 1.51x sequential.\n',
     L3_EN_INSERT),
]


def main(dry: bool) -> int:
    ok = True
    for name, old, new in EDITS:
        path = os.path.join(DOCS, name)
        text = open(path, encoding='utf-8').read()
        n = text.count(old)
        if n != 1:
            print('ABORT %-24s anchor=%d' % (name, n))
            ok = False
            continue
        if not dry:
            open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %-24s +%d chars' % (name, len(new) - len(old)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main('--dry' in sys.argv))
