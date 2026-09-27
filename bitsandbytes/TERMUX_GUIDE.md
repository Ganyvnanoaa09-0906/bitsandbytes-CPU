# Termux（Android/ARM64）构建与验证指南

本文档是 `build_termux.sh` 的使用与验证流程。文件名用 `.md` 而不是中文名，
避免跨平台编码问题；内容本身是 UTF-8（bash 与编辑器都能正确读取）。

---

## 一、为什么需要单独的 `build_termux.sh`

`build_linux.sh` 在 Termux 上会在**四处**踩到，所以没有直接复用：

| # | Termux 与普通 Linux 的差异 | 后果 |
|---|---|---|
| 1 | 编译器是 `clang++`（`pkg install clang`），**没有 `g++`** | `build_linux.sh` 默认 `g++`，直接失败 |
| 2 | OpenMP 需要单独 `pkg install libomp` | 没装 `-fopenmp` 会链接失败 |
| 3 | `-march=native` 在 Android/Termux 的 clang 上行为不稳 | 可能生成宿主不支持的指令 |
| 4 | 磁盘/内存小 | 不宜生成多余中间产物 |

**代码侧的 ARM 支持本来是就绪的**，不需要改内核：
`csrc/cpu_ops.cpp` 有 4 处 `#if defined(_M_ARM64) || defined(__aarch64__)` 分支，
包含 `<arm_neon.h>`，并使用 ARM64 的原生 FP16 转换。缺的只是**构建入口**。

> **历史更正**：`build_termux.sh` **从未进过版本控制**
> （`git log --all --follow -- build_termux*` 无任何记录，全盘搜索也没有）。
> 所以本脚本是**新写的**，不是"恢复"出来的。早期文档（`Termux验证流程.txt`）
> 提到过它，但那份文档本身是仓库里唯一存在过的 Termux 资产。

---

## 二、手机侧准备

1. 安装 Termux —— **用 F-Droid 或 GitHub releases，不要用 Play 商店版**（太旧）
2. 打开一次 Termux 让它完成初始化
3. 装依赖：

```bash
pkg update
pkg install -y clang libomp make
```

`libomp` 只影响 OpenMP 并行；没装也能构建，脚本会自动降级为单线程并明确提示。

---

## 三、两种运行方式

### 方式 A：自动（从 PC 用 ADB 驱动）

```powershell
# 需要先让 adb 能看到手机（USB 调试 / HDB / 或无线调试）
powershell -NoProfile -ExecutionPolicy Bypass -File D:\work\termux_adb_test.ps1
```

可选参数：

| 参数 | 作用 |
|---|---|
| `-ProbeOnly` | 只探测设备与 Termux 环境，不构建 |
| `-SkipSelftest` | 只构建，不跑 C 层自检 |
| `-WaitForDeviceSec 60` | 等手机上线最多 60 秒 |
| `-Adb <路径>` | 手动指定 adb（默认自动挑选版本最高的那个） |

脚本会自动：
1. 挑选可用的 adb（本机实测：桌面那个是 2013 年的 `1.0.26`，行为不可靠；
   MuMu 自带的是 `1.0.41 / 36.0.0`，正确工作）
2. 探测手机状态，并**区分** `device` / `offline` / `unauthorized`（三者的修法完全不同）
3. 只打包构建真正需要的约 300 KB（整仓 10 MB 里大部分是文档与 CUDA 头）
4. 通过 **`/sdcard/Download`（共享存储）** 中转，然后交给 Termux 一行命令取用
5. 校验设备侧文件大小与本地一致后才继续（只说"传完了"不够）
6. 给出**一行**可在 Termux 里直接粘贴的命令（解包 + 构建 + 自检）

**为什么必须走 `/sdcard` 而不是直接写进 Termux 家目录**（实测，含一次被证伪的推断）：

- `adb shell` 的身份是 `uid=2000(shell)`，而 `/data/data/com.termux/`
  对它返回 **`Permission denied`**（Android 10 的沙箱）。
- 因此 **`adb push` 写进 Termux 家目录会失败**，**`adb shell ... > 文件` 的管道同样会失败**
  —— 我最初以为"管道是以 Termux 身份运行的所以能写"，这是**错的**，两个都不行。
- `/sdcard` 是共享存储：`shell` 用户可写，Termux 也能读，而且**不需要**
  `termux-setup-storage`。
- 构建本身**无法**通过 adb 在 Termux 内执行：Termux 有自己的 keystore，
  外部进程无法在里面跑命令（除非开放 RUN_COMMAND 权限）。所以最后一步必然是
  用户在 Termux 里粘贴一行。

**设备侧命令的兼容性约束**（Android 10 实测）：
`/system/bin/sh` 不支持 `if ...; then ...; else ...; fi`（mksh 报 `unexpected 'else'`）；
toybox 的 `stat` 没有 `-c`/`--format`（`stat -c %s` 报 `Needs 1 argument`）。
取文件大小用 `wc -c < file` 可用。

### 方式 B：手动（不依赖 ADB）

把仓库传到手机上（git clone，或用任何方式拷贝），然后：

```bash
cd <仓库目录>
bash build_termux.sh --selftest
```

---

## 四、验收判据（三条都要满足）

1. `build_termux.sh` 打印出 **`DONE.`**
2. 导出符号抽查里 **5 个关键符号全部 `[OK]`**
   （`cquantize_blockwise_cpu_fp32` / `cdequantize_blockwise_cpu_fp32` /
   `cgemm_8bit_inference_cpu_fp32` / `cgemv_4bit_inference_cpu_fp32` /
   `coptimizer_update_8bit_blockwise_cpu`）
   —— 逐个断言而不是数总数：总数对得上也可能缺关键项
3. 自检打印 **`[PASSED]`**

> **关于自检项数（更正旧文档）**：早期文档写"5 项（含 `gdn_fwd`）"，
> 但 `selftest_cpu.c` 里实际只有 **4 项**：
> ①quantize 往返 ②gemm_8bit ③gemv_4bit ④AdamW8bit。
> 脚本与本文档以**代码为准**。

---

## 五、边界（不夸大）

- Termux **通常装不上 torch**，所以验证到 **C 层自检**为止 ——
  验证的是**内核数值正确性**，不是训练能力。
- 自检只覆盖 4 个内核。NEON 路径与标量路径的**性能**差异未测
  （判据只保证数值正确）。
- 手机型号/SoC 不同，`-mcpu=native` 的探测结果会不同；脚本会自动降级。
- `termux_adb_test.ps1` 是**纯 ASCII** 的（含所有提示信息），这是刻意的：
  Windows PowerShell 5.1 按系统 ANSI 代码页读 `.ps1`，UTF-8 无 BOM 的中文会被
  误解码，**不只是显示乱码，而是直接破坏语法解析**（实测：一个在 UTF-8 下
  完全合法的文件报了 6 个语法错误）。中文说明放在本文件里，它不参与解析。

## 六、相关文件

- `build_termux.sh` —— Termux 构建入口（新写）
- `selftest_cpu.c` —— 无 torch 的 C 层自检（4 项）
- `build_linux.sh` —— 普通 Linux 构建入口（x86_64 AVX2 / aarch64 NEON）
- `D:\work\termux_adb_test.ps1` —— ADB 自动化验证台
- 报告 `CPU_FORGE_TECHNICAL_REPORT.md` 相关章节
