# ISA 兼容矩阵（2026-10-19 回归）

本文件由 `isa_matrix.sh` 的实测输出整理；原始日志见 `bench_results/isa_matrix_raw.log`，
构建脚本 `research/isa_build.py`（zig 交叉编译）。复现（x86 开发机 + WSL）：

```bash
python D:\work\research\isa_build.py                 # 编译全部变体（Windows，zig）
wsl -d Ubuntu -- bash /mnt/d/work/bitsandbytes-CPU/isa_matrix.sh   # 审计 + 运行时自检
```

## 构建矩阵

| 目标 | 工具链 | 产物 | glibc 需求 | 用途 |
|---|---|---|---|---|
| x86_64 AVX2 | zig 0.14 clang | `libbitsandbytes_cpu.so` | ≤2.16 | manylinux_2_17 wheel（serial，无 OpenMP） |
| x86_64 generic | zig 0.14 clang | `libbitsandbytes_cpu_generic.so` | ≤2.16 | 无 AVX2 x86 的 portable 回退 |
| x86_64 AVX-512 | zig 0.14 clang | `libbitsandbytes_cpu_avx512.so` | ≤2.16 | AVX-512/BF16 机器加速（EPYC 实测） |
| x86_64 原生 OpenMP | 本机 g++ 15 | manylinux_2_39 wheel | ≤2.39 | 新 glibc 的并行版（auditwheel 打包 libgomp） |
| aarch64 gnu | zig 0.14 clang | `build_arm64/libbitsandbytes_cpu.so` | ≤2.27 | 源码/自编 Linux ARM |
| aarch64 musl | zig 0.14 clang | `build_arm64/libbitsandbytes_cpu_musl.so` | 无 | Termux/静态 musl |
| aarch64 静态自检 | zig 0.14 clang | `build_arm64/selftest_arm64_musl` | 无 | qemu-aarch64 运行时验证 |

## 静态指令审计

| 变体 | AVX 类指令 | zmm（AVX-512） | BF16 指令 | 导出 `has_avx512bf16_cpu` |
|---|---|---|---|---|
| x86_64 AVX2 | 8,058 | 0 | 0 | 否 |
| x86_64 generic | **0** | 0 | 0 | 否 |
| x86_64 AVX-512 | 9,890 | 2,947 | **182** | 是 |
| aarch64 gnu/musl | —（ARM） | — | — | 否（NEON 类指令 570 条，AArch64 机器码） |

结论：generic 库**零 AVX 指令**（可在任何 x86_64 上跑）；AVX2 库无 AVX-512/zmm；
AVX-512 库的 182 条 BF16 指令与 `_cpu_caps` 的 `avx512_bf16` 门控一致（不会在无 BF16
的机器上误载入）。

## 运行时自检（C 内核独立自检，5 项：quantize 往返 / gemm_8bit / gemv_4bit /
AdamW8bit / 向量-标量叉验证）

| 变体 | 运行方式 | 结果 |
|---|---|---|
| x86_64 AVX2 | 本机（R5-4500U，AVX2） | **5/5 PASS** |
| x86_64 generic | 本机 | **5/5 PASS** |
| x86_64 AVX-512 | 本机无 AVX-512 → 静态审计；EPYC 9Y24 容器真机 | 静态 182 BF16 指令；EPYC 上 **5/5 PASS**、GDN 5/5（见 `EPYC_VALIDATION.md`） |
| aarch64 musl | qemu-aarch64（x86 开发机） | **5/5 PASS** |

## 边界

- aarch64 是**架构级**验证（qemu + 静态 musl），不是 Android/bionic 真机；Termux 真机
  历史结论仍有效。
- aarch64 gnu 库当前 zig 默认 glibc 目标为 2.27（若将来发 aarch64 manylinux wheel，
  需显式 `-target aarch64-linux-gnu.2.17`）。
- AVX-512 运行时验证依赖真实硬件（EPYC 容器），本机只能做静态/编译检查。
