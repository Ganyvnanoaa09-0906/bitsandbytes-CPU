# Linux wheel 平台策略与 old-glibc 并行 wheel 调研

日期：2026-10-19

## 现状（0.50.2.dev4 发布形态）

| wheel | 构建方式 | 并行 | 适用 |
|---|---|---|---|
| `manylinux_2_17_x86_64`（portable） | zig 交叉编译，glibc 2.17 目标，无 OpenMP | **serial** | glibc ≥ 2.17 的所有 x86_64（Ubuntu 18.04+/Debian 10+/CentOS 7+） |
| `manylinux_2_39_x86_64`（native） | 本机 g++ 15 + OpenMP（auditwheel 打包 libgomp） | OpenMP | glibc ≥ 2.39（Ubuntu 24.04+/Debian 13+） |
| `win_amd64` | MSVC（历史构建） | OpenMP（vcomp140） | Windows |
| sdist | 源码，Linux 上现场 `build_linux.sh` | OpenMP（本地 g++） | 有编译器的 Linux |

**缺口**：Ubuntu 22.04（glibc 2.35）、Debian 12（2.36）这类系统只能拿到 serial 版，
多核机器会损失 OpenMP 并行度（EPYC 容器实测时发现）。

## 调研：用 CentOS 7 的 libgomp 做 manylinux_2_17 + OpenMP

思路：把 CentOS 7（glibc 2.17）的 `libgomp.so.1` 随 wheel 打包（scikit-learn 等 wheel
也这么做），再用 auditwheel 打 `manylinux_2_17` 标签。

已验证：
- CentOS 7.9 `libgomp-4.8.5-44.el7.x86_64.rpm` 可下载（aliyun centos-vault）；
- 解出的 `libgomp.so.1.0.0`（154,840 B，md5 `4b6cd4878e31bcd05f23ffd40c8edf80`）
  NEEDED 只有 `libpthread/libc`，**max GLIBC = 2.17**，SONAME `libgomp.so.1`，可直接 bundle；
- 本地原型：g++ 15 `-fopenmp` 编译 + 链接该旧 libgomp 成功，`ldd` 正确解析。

**阻塞点**：宿主 g++ 15 的静态 libstdc++/libgcc 把 GLIBC 需求拉到 2.34–2.38
（`pthread_once@2.34`、`arc4random@2.36`、`_dl_find_object@2.35`、`__isoc23_strtoul@2.38`），
auditwheel 仍会判 manylinux_2_39。zig clang 能交叉编译到 glibc 2.17，但它不支持
libgomp ABI（`-fopenmp` 生成 `__kmpc_*` = libomp ABI，`-fopenmp=libgomp` 被忽略）。

## 可选后续方案（按工作量）

1. **devtoolset-8 工具链**（CentOS 7 的 GCC 8 + glibc 2.17）：用它编 `.so` 并 bundle
   devtoolset 的 libstdc++/libgcc/libgomp；需要下载若干 RPM，可行性较高。
2. **自写 pthread 并行层**替换 OpenMP pragma（37 处）：zig 交叉编译到 2.17 + libc++ 静态，
   完全无外部运行时依赖；工作量中等，但可顺带摆脱 libgomp 许可/体积问题。
3. **容器内构建 manylinux_2_35 wheel**：在 Ubuntu 22.04 容器（豆包 EPYC）上构建
   `manylinux_2_35` OpenMP wheel，覆盖 Ubuntu 22.04/24.04、Debian 12、RHEL 9；
   需要把产物从容器传回本机（文件下载或 base64）。
4. 维持现状 + 文档说明：老 glibc 用户用 serial wheel 或 sdist 自编（sdist 默认就带 OpenMP）。

## 建议

短期发布维持现状并在 README 写明；中期优先方案 1 或 3（方案 3 无需改代码，只需传输）。
方案 2 适合作为"彻底解决 + 去掉 libgomp 依赖"的专项。
