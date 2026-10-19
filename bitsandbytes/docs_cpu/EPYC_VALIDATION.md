# EPYC 容器 ISA 验证记录（2026-10-19）

对象：AMD EPYC 9Y24（Zen5，容器内 2 vCPU / 3.9GB RAM）
系统：Ubuntu 22.04.5，内核 6.6.95（容器 sandbox），g++ 11.4，Python 3.12.11，torch 2.14.0+cpu
产物：`bitsandbytes-cpu-fork==0.50.2.dev4`（PyPI）

## 1. CPU 指令集（内核暴露）

```
avx avx2 fma f16c sse4_2
avx512f avx512bw avx512dq avx512vl avx512_bf16 avx512_vnni
avx512vbmi avx512_vbmi2 avx512_ifma avx512cd avx512bitalg avx512_vpopcntdq
（无 AMX；avx512vnni 的写法是 avx512_vnni，带下划线）
```

## 2. 第一轮：sdist 构建矩阵（`container_isa_report.sh`）

| 变体 | 结果 |
|---|---|
| AVX2 base | 编译 OK，C 自检 5/5 PASS |
| generic x86-64 | 编译 OK，0 条 AVX 指令，C 自检 5/5 PASS |
| AVX-512 | 编译 OK（280KB），**182 条 BF16 指令**，C 自检 **5/5 PASS（真机执行）**，ctypes 加载 OK |

`pip install bitsandbytes-cpu-fork==0.50.2.dev4` 在 glibc 2.35 上正确选中
`manylinux2014_x86_64.manylinux_2_17_x86_64` wheel（可移植 serial 版），三个 `.so` 齐全。

## 3. 第二轮：已发布 wheel 的运行时加载（`container_isa_runtime_check.sh`）

| 检查项 | 结果 |
|---|---|
| `avx2_supported()` / `avx512_supported()` | True / True |
| `LOADED_NATIVE_LIB` | `.../libbitsandbytes_cpu_avx512.so` ✅ |
| `has_avx512bf16_cpu` 符号 | 存在且 `has_avx512bf16_cpu() -> 1` ✅ |
| 8-bit 量化往返 | max abs err 0.0288（正常量化误差） |
| `AdamW8bit` 单步 | finite，max delta 1.05e-3 |
| 2048³ fp32 matmul | 0.093 s → **185.3 GFLOPS**（2 vCPU），无 SIGILL |
| GDN 内核自检（fp32/bf16/fp16 fwd+bwd） | **selftest PASSED**（相对误差均在容差内） |

这闭环了发布前的两个高危验证点：
1. AVX-512 产物确实含 BF16 指令 → 加载条件必须包含 `avx512_bf16`；EPYC 9Y24 有该标志，
   实际选中 AVX-512 库并正常执行；
2. AVX-512 库不是"能编译但不能跑"——真机上 8bit 优化器/量化/GEMM/GDN 全部正常。

## 4. 已知事项 / 后续

- **serial wheel**：Ubuntu 22.04（glibc 2.35）只能拿到 `manylinux_2_17` portable 版（无
  OpenMP）；多核 EPYC 容器会损失 OpenMP 并行度。下一版可考虑补一个"老 glibc + OpenMP"wheel。
- **VNNI 未用于选择**：当前加载条件只要求 `avx512f/bw/dq/vl + avx512_bf16`；VNNI 代码路径
  未使用，因此不需要在条件里检查 `avx512_vnni`。
- **`kernels` 可选包告警**：第二轮输出里的 `Failed to load CPU gemm_4bit_forward from
  kernels-community: No module named 'kernels'` 是上游可选 hub-kernel 探测；已在
  `backends/cpu/ops.py` 里改为：未安装可选包时仅 debug 日志，不再向 CPU-only 用户刷 warning。
- 该容器 `/tmp` 会在任务间清空；本记录即归档（原始输出已由操作者回传）。

## 5. 复现命令（任意 Linux）

```bash
cd /tmp
bash container_isa_report.sh 2>&1 | tee /tmp/cpu_forge_isa_report.txt
bash container_isa_runtime_check.sh 2>&1 | tee /tmp/cpu_forge_runtime_report.txt
```

## 2026-10-19 补充：manylinux_2_17 并行 wheel 已可构建

之前记录"EPYC 上 pip 选到 manylinux_2_17 serial wheel"的缺口已解决：`build_oldgomp_linux.sh`
用 CentOS7 devtoolset-8 产出带 OpenMP 的 manylinux_2_17 wheel（C 自检 9/9，优化器内核比 serial
快 4×）。注意该 wheel 的 avx512 .so 仍是 zig serial；在 AVX-512 机器上比较 AVX2 OpenMP 与
AVX-512 serial 时可用 `BNB_CPU_DISABLE_AVX512=1` 切换。详见 `bench_results/oldglibc_openmp_wheel.md`。

