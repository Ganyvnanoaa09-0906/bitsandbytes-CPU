# manylinux_2_17 + OpenMP wheel（老 glibc 并行版）——2026-10-19

## 背景

zig 交叉编译的 `manylinux_2_17` wheel 是 **serial** 的（zig clang 不支持 libgomp ABI），
Ubuntu 18.04–22.04 / Debian 10–12 / RHEL 7–9 用户只能拿到无并行版本。本轮用 **CentOS 7
devtoolset-8**（GCC 8.3.1，glibc 2.17 sysroot）在本地编译出并行 .so，链接旧 libgomp、
静态化 libstdc++/libgcc，auditwheel 打回 manylinux_2_17 并 bundle libgomp。

## 工具链（无需 root）

- CentOS Vault SCLo repo 的 devtoolset-8 RPM（gcc/g++/libstdc++-devel/binutils）+
  CentOS7 os repo 的 glibc-devel/glibc-headers/kernel-headers/mpfr/libmpc/gmp/glibc/libgomp
  RPM；
- 用 repodata/primary.xml.gz 解析文件名 → curl 下载 → `rpm2cpio | cpio -idm` 解到
  `$HOME/dts8/root`（C 库/头文件）、devtoolset-8 在 `.../opt/rh/devtoolset-8/root`；
- 编译时 `--sysroot=$HOME/dts8/root -B<sysroot>/usr/lib64/` 即可用旧头文件/旧 libgomp/
  旧 crt 对象。devtoolset 的 cc1plus 额外需要 libmpfr.so.4/libmpc.so.3/libgmp.so.10
  （从 CentOS7 RPM 解出即可）。

## 产物

| 文件 | glibc 需求 | NEEDED | 说明 |
|---|---|---|---|
| `libbitsandbytes_cpu_generic.so` | ≤2.14 | libm/libgomp/libpthread/libc/ld | 无 AVX 的 portable 版，**带 OpenMP** |
| `libbitsandbytes_cpu.so`（AVX2） | ≤2.14 | 同上 | Haswell+ 主路径，**带 OpenMP** |
| `libbitsandbytes_cpu_avx512.so` | — | — | 仍是 zig **serial** 版（见下） |
| wheel（repair 后） | — | 1.62 MB | tag `manylinux2014_x86_64.manylinux_2_17_x86_64` |
| bundle `libgomp-a34b3233.so.1.0.0` | **≤2.17** | — | auditwheel 自动打包，OMP_4.5 |

## 验证（WSL，glibc 2.43 上安装）

- `pip install` 后 `load_native()` OK；C 自检 **9/9 PASS**（直接链接 wheel 内 AVX2 .so）；
  torch GDN 自检 PASSED；
- OpenMP 并行确实生效（同一环境对比 17M 参数 AdamW8bit 单步）：
  **zig serial manylinux_2_17 wheel：176.9 ms → devtoolset-8 OpenMP wheel：44.2 ms（4.0×）**；
- 256MB copy 16.7 GB/s（torch，仅参考）。

## 限制 / 后续

- wheel 内 `libbitsandbytes_cpu_avx512.so` 仍是 zig **serial** 版（zig 不能产生 libgomp ABI
  调用；CentOS7 GCC 8.3 不支持 AVX512-BF16 intrinsics）。AVX-512 机器（如 EPYC 9Y24）
  会默认选它 → 无并行；临时方案：`BNB_CPU_DISABLE_AVX512=1` 强制 AVX2 OpenMP 路径。
  后续可用更新工具链（devtoolset-11+ 或 manylinux_2_28 容器）补 AVX-512 OpenMP 版本。
- 该 wheel 尚未发布：等待 dev6 发版（需要 PyPI token）。构建脚本已进仓库：
  `bitsandbytes/build_oldgomp_linux.sh`（`BNB_DTS_ROOT` 指向解压好的 CentOS7 root）。
- `cpu_features.h`/`cpu_cache.h` 的 `<cpuid.h>` 双包含问题（GCC 8 的 cpuid.h 无 include guard）
  已用 `BNB_CPUID_H_INCLUDED` 哨兵修复——这是老工具链能编过的前提，也兼容新编译器。
