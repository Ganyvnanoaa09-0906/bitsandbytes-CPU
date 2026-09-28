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

### 自检现在共 5 项

| # | 检查 | 说明 |
|---|---|---|
| 1 | quantize_blockwise 8bit 往返 | 误差 < 0.6% |
| 2 | gemm_8bit 前向 | 与参考近似 |
| 3 | gemv_4bit(nf4) | 输出有限且非全零 |
| 4 | AdamW8bit 单步更新 | 参数变化合理 |
| 5 | **向量分派 × 标量叉验证** | 同一入口、逐元素等价的输入，只改 `blocksize`，使一个走向量分派、一个走标量回退，要求输出**逐位相同** |

> **第 5 项为什么必须有**：前 4 项**不能**证明向量路径正确。实测第 3 项用的是
> `bs=2`，而 NEON 分派的条件是 `n % blocksize == 0 && blocksize % 16 == 0`
> （`cpu_ops.cpp:1265`）⇒ 第 1/2/3 项在 ARM64 上**全部跳过 NEON**；
> `neon_absmax` 与 bf16/fp16 转换路径更是从未被覆盖。
> 只看"4/4 通过"就宣称 ARM64 支持，是在主张并不存在的覆盖。
> 第 5 项与架构无关，因此在 x86 与 ARM64 上跑的是**同一条判据**。

**实测结果（两个工具链都跑过）**：

| 工具链 | 第 5 项 | 含义 |
|---|---|---|
| MSVC / AVX2 | **PASS** | AVX2 分派 == 标量，逐位一致 |
| clang / NEON | **PASS** | NEON 分派 == 标量，逐位一致 |

> ⚠️ 第 5 项的**第一版判据是错的**，被 MSVC 当场抓住（128/128 不一致、最大差 1.46）：
> 那一版只让**缩放值的集合**重复出现，却让一个布局每行 3 个 scale、另一个每行 6 个，
> 于是**逐 k 的 scale 序列不同** ⇒ 两次调用本是不同的算式。
> 修正为让细粒度布局的第 `b` 块取 `seq[b/2]`，使每个元素上的缩放完全一致。
> 教训与本文档其余部分一致：**判据要落在逐元素上，不能只看形式对齐**。

> **关于自检项数的历史更正**：早期文档写"5 项（含 `gdn_fwd`）"，
> 而 `selftest_cpu.c` 里当时**只有 4 项且没有 GDN**（曾把第 3、4 项顺序标反）。
> 现在确实是 5 项，但第 5 项是上面的叉验证，仍然**不是** GDN。
> 以代码为准 —— 这类"文档说有一项、代码里没有"的差异在本项目出现过多次。

---

## 五、边界（不夸大）

- Termux **可以装 torch**（更正：早期文档写"通常装不上"是错的）。PyPI 上确实每个
  aarch64 轮子都是 `manylinux_2_28`（glibc），而 Android 用 bionic，**pip 路线走不通**；
  但 Termux 仓库有社区移植包：
  ```
  python-torch 2.11.0-2   下载 35.5 MB   安装 266 MB
  ```
  实测在 Android 10 / aarch64 / Python 3.14.6 上装成并可 import。
  同理 numpy/psutil 也必须走 `pkg install python-numpy python-psutil` ——
  `pip install numpy` 找不到 Android 轮子，会转去源码构建然后失败。
- 但装得上 torch **不等于**能在同一进程里同时用 torch 和本项目的 `.so`：
  `build_termux.sh` 默认 `-fopenmp`，**静态链接 libomp**；torch 自带另一份。
  一个进程里两份 OpenMP 运行时，LLVM 的 libomp 会**主动 abort**：
  ```
  OMP: Error #15: Initializing libomp.a, but found libomp.a already initialized.
  Aborted (core dumped)     # exit 134 / SIGABRT
  ```
  它这么做是对的 —— 宁可不跑也不冒算错的风险。
  **验收结论：这不是内核缺陷，是构建配置。** 用 `--no-openmp` 重建单线程版即可解除，
  实测 `torch + bitsandbytes` 同进程可正常跑通（见下）。
  **不要**用 `KMP_DUPLICATE_LIB_OK=TRUE` 绕过 —— 它的官方文档写明
  "may cause crashes or **silently produce incorrect results**"。
  一个专门验数值正确性的套件，绝不能跑在一个允许静默错误结果的开关下；
  那等于把判据换成"只要不崩就算过"。
- 因此 Python 侧检查必须**按 OpenMP 域分进程**（实测得出的结构）：
  | 进程 | 加载什么 | 跑什么 |
  |---|---|---|
  | `termux_check.py` | 只加载 `.so`（ctypes） | T1–T5，**从不 import torch** |
  | `torch_part.py` | 只 import torch | T6/T7（`disk_balancer`、`latent_chunk_store`） |
  | `torch_part.py --with-bnb` | torch **且** `.so` | T8（`AdamW8bit` 真跑一步） |
  父进程连"探测 torch 是否存在"都用子进程做 —— 那一步本身就是崩溃点。
  T8 单独一个进程是刻意设计：它是唯一**预期**会 abort 的一步，
  结论不能取决于崩溃落在文件里的哪一行。（不用 `fork()`：torch 那时已起线程，
  fork 会把锁住的互斥量复制给子进程而导致死锁。）
- 自检覆盖 5 项，其中第 5 项覆盖了 4-bit GEMV 的**向量分派**（NEON/AVX2 与标量逐位一致）。
  **仍未覆盖**：`neon_absmax`、`neon_f32_to_bf16x4/fp16x4` 等被 `__aarch64__` 守卫的
  辅助函数**从未在任何一项里被走到** —— 它们目前只有"能编译过"这一层证据。
  若要宣称完整，需要再加针对这些路径的用例。
- NEON 与标量路径的**性能**差异未测（第 5 项只保证数值正确）。
- 手机型号/SoC 不同，`-mcpu=native` 的探测结果会不同；脚本会自动降级。
- **完整回归套件（`run_all_tests.py`）在 Termux 上跑不了**：它的多数子脚本在模块层
  `import torch`，同时又加载 `.so`，正是上面那个双 OpenMP 冲突。长训练在 x86 上验
  （R5 与 i5，各 1000 步）。
- `termux_adb_test.ps1` 是**纯 ASCII** 的（含所有提示信息），这是刻意的：
  Windows PowerShell 5.1 按系统 ANSI 代码页读 `.ps1`，UTF-8 无 BOM 的中文会被
  误解码，**不只是显示乱码，而是直接破坏语法解析**（实测：一个在 UTF-8 下
  完全合法的文件报了 6 个语法错误）。中文说明放在本文件里，它不参与解析。

### 实测结论（SPN-AL00 / Android 10 / aarch64 / Python 3.14.6 / torch 2.11.0）

| 项 | 结果 |
|---|---|
| ARM64 构建（静态 libomp，1.1 MB） | 成功，23 个 CPU 符号，5 个关键符号全在 |
| C 层自检 5 项 | 0 失败（含 NEON 与标量**逐位一致**） |
| ctypes 域 T1–T5 | 5/5 通过（`e_machine=0xB7`；8bit 往返 0.392%；`gemm_8bit` 对 float64 独立参考 2.514e-07） |
| torch 域 T6/T7 | 2/2 通过 |
| torch + bitsandbytes 同进程 T8（多线程 `.so`） | **SIGABRT**，双 OpenMP —— 记为 SKIP 并给出原因与修法 |
| torch + bitsandbytes 同进程 T8（`--no-openmp`，128 KB） | **8/8 通过**；`AdamW8bit` 真跑一步：`max|dW| 6.2269e-02`，loss `0.3553 → 0.0296` |
| 单线程版数值一致性 | 与多线程版**完全相同**（T4 0.392%、T5 2.514e-07） |

`gemm_8bit` 那条参考值值得一提：C 自检把 `gemm_8bit` 与它自己的标量路径对比，
那是**一致性**检查 —— 两条路径共有的错误也会通过。T5 的参考值是**本脚本用 numpy
在 float64 下独立算出来的**（权重字节由脚本自己反量化），所以是独立判据。

## 六、曾经踩过的坑（都已修，记下来避免重犯）

| 坑 | 表现 | 真因 |
|---|---|---|
| `adb push` 进 Termux 家目录 | permission denied | `adb shell` 是 `uid=2000(shell)`，`/data/data/com.termux/` 对它不可读 |
| 用管道写 `/data/data/com.termux/...` | 同样失败 | 管道也以 shell 身份运行 —— 我曾错误推断"管道以 Termux 身份运行" |
| 用 `test -x` 探测 Termux 是否安装 | 明明装了却报"未安装" | 该测试因权限失败；应查 `pm list packages` |
| 共享存储 `/sdcard` 给 Termux 用 | Termux 读不到 | **两个独立原因**：① 需 `termux-setup-storage` 授权；② 即使授权了，**adb 在 `/sdcard` 创建的文件是 `root:sdcard_rw` 0660，Termux 的 uid 不在该组**，`cp` 照样 Permission denied —— 且 `/sdcard` 是 FUSE，chmod 无效。结论：**别用 `/sdcard` 传文件给 Termux，用 `/data/local/tmp`**（0777，adb 可写，Termux 可读） |
| 设备侧 `if ... else ... fi` | `unexpected 'else'` | Android 的 `/system/bin/sh` 是 mksh |
| 设备侧 `stat -c %s` | `Needs 1 argument` | Android 10 的 toybox `stat` 不支持 `-c/--format`；用 `wc -c` |
| 让用户 `bash /sdcard/.../x.sh` | `Permission denied` | `/sdcard` 是 FUSE，**所有文件都被挂成不可执行**；`bash 路径` 仍需读权限，而它连读都不让。先 `cp` 到家目录 |
| 脚本第一行 `exec > "$LOG"` | 用户屏幕上**一片空白**，看不出死活 | 经 `curl ... \| bash` 执行时 `$HOME` 为空，日志路径变成 `/bnb_termux.log`，写根目录被拒 → bash 在打印任何字符前退出。**手工运行的脚本绝不能静默** |
| `curl \| bash` 卡住无输出 | 一直等 | 我这边的 HTTP 服务随会话后台 job 一起消失了（`adb` daemon 也重启过）。**传输层的失败会被误读成被测代码的问题** —— 改成把整包 base64 内嵌进单个 `.sh`，网络彻底不在链路里 |
| 生成的脚本 here-doc 起始符与首个 payload 行粘连 | `here-document at line N delimited by end-of-file` | PowerShell here-string `@'...'@` **会吃掉结尾换行**，于是 `<<'__B64_EOF__'` 与第一行 base64 拼成一行。**而且 `bash -n` 只警告、仍退出 0** —— 单看退出码发现不了。改为"stderr 有任何输出即判失败" + 对 payload 做 sha256 硬校验 |
| AVX2 代码在 ARM64 上被编译 | `unknown type name '__m128i'` | 两整组 AVX2 函数**完全没有守卫**（`cpu_ops.cpp`） |
| 标量助手报 undeclared | `use of undeclared identifier` | `scalar_absmax` / `scalar_gemv_4bit_inference` 被关在 AVX2 守卫内，却被无守卫的回退路径调用 |
| clang 警告"treating 'c' input as 'c++'" | 编译警告 | `selftest_cpu.c` 必须按 C++ 编译（内部有 `extern "C"`），已显式加 `-x c++`，对应 MSVC 的 `/TP` |
| 手工挑文件打发布包 | `fatal error: 'common.h' file not found` | 我按名字挑了 6 个文件，漏了 `common.h` —— 而构建把 `csrc/` 当 include 根整体读。**正确做法：整目录拷贝**，并加脚本穷举所有 `#include "..."` 逐个校验 |
| `disk_balancer.py` 的 docstring 里写 `C:\cache` | Python 3.14: `SyntaxWarning: "\c" is an invalid escape sequence` | 非 raw docstring 里的 Windows 路径。3.11 只警告所以长期没暴露，3.14 明确说将来会失效（届时**直接导入失败**）。已改 raw docstring；并写 `scan_escapes.py` 全仓扫描，另抓出 `bitsandbytes/tools/verify_train_release.py` 同类问题 |

## 七、相关文件

- `build_termux.sh` —— Termux 构建入口
- `selftest_cpu.c` —— 无 torch 的 C 层自检（5 项）
- `build_linux.sh` —— 普通 Linux 构建入口（x86_64 AVX2 / aarch64 NEON）
- `pythonInterface.cpp` —— ctypes 导出签名（写 ctypes 调用**必须**照它，不能照名字猜）
- `D:\work\bitsandbytes-CPU\termux_check.py` —— ctypes 域检查 T1–T5（从不 import torch）
- `D:\work\bitsandbytes-CPU\torch_part.py` —— torch 域检查（`--with-bnb` 时才加载 `.so`）
- `D:\work\bitsandbytes-CPU\i5build\build_termux_bundle.ps1` —— 打发布包（整目录 + include 校验）
- `D:\work\bitsandbytes-CPU\i5build\make_selfcontained.ps1` —— 生成 base64 内嵌的自包含脚本
- `D:\work\bitsandbytes-CPU\i5build\verify_selfcontained.py` —— payload sha256 往返校验
- `D:\work\bitsandbytes-CPU\i5build\setup_termux_test.sh` —— 设备侧一键验收（经 HTTP/ADB 隧道）
- `D:\work\termux_adb_test.ps1` —— ADB 自动化验证台
- 报告 `CPU_FORGE_TECHNICAL_REPORT.md` 相关章节
