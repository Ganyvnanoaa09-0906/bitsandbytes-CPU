# 给豆包：EPYC 容器上的 CPU 指令集兼容性验证

辛苦啦～我们在做一个**纯 CPU 训练**的 bitsandbytes fork（项目名 `bitsandbytes-cpu-fork`，已经发到 PyPI）。现在需要在她那台 **EPYC 容器**上验证"各种指令集全兼容"这件事，帮我们跑一份环境+构建+自检报告就行。

## 推荐方式：跑一个脚本（约 5–15 分钟）

1. 把 **`container_isa_report.sh`**（和本文件同一目录）复制到容器里任意位置，比如 `/tmp/`。
2. 在容器里执行：

```bash
cd /tmp
bash container_isa_report.sh 2>&1 | tee /tmp/cpu_forge_isa_report.txt
```

3. 把终端输出（或 `/tmp/cpu_forge_isa_report.txt` 的内容）**整段贴回来**。

脚本会依次做这些事：

- 打印 CPU 型号、内核识别到的全部 `avx2 / fma / f16c / avx512f / avx512bw / avx512dq / avx512vl / avx512_bf16 / avx512vnni / amx` 标志；
- 检查 `g++ / gcc / clang++ / python3 / pip / venv / torch / transformers` 版本；
- 探测 PyPI 和 hf-mirror 网络；
- 在 `/tmp` 建独立 venv，装 `bitsandbytes-cpu-fork==0.50.2.dev4`，跑它不带 torch 的 CLI（`version / detect / doctor`）；
- 从 PyPI 下载 sdist，用 `build_linux.sh` **现场编译三种产物**：AVX2（CPU 不支持时自动降级）、generic x86-64、以及（若 CPU 有 avx512f）AVX-512 变体，并跑每种的 C 自检；
- 如果容器有 torch，再 import bitsandbytes 看**实际选中哪个 native 库**，并跑 Gated-DeltaNet 内核自检。

**安全说明**：脚本只往 `/tmp` 写（venv、源码、产物、报告），不需要 sudo，不碰系统文件和其它目录，可以放心跑。

### 如果容器没网 / 不想联网

先只做环境探测（不下载不编译，约 30 秒）：

```bash
bash container_isa_report.sh --probe-only 2>&1 | tee /tmp/cpu_forge_isa_report.txt
```

如果你们能把 sdist 传进容器，可以离线跑构建矩阵：

```bash
BNB_SRC_TARBALL=/path/to/bitsandbytes_cpu_fork-0.50.2.dev4.tar.gz \
    bash container_isa_report.sh --offline 2>&1 | tee /tmp/cpu_forge_isa_report.txt
```

## 备用方式：不跑脚本，直接回答下面 8 个问题

1. **CPU 型号**：`grep -m1 'model name' /proc/cpuinfo` 或 `lscpu | grep 'Model name'`
2. **AVX-512 标志**：`grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^avx(2|512)' | sort | tr '\n' ' '`
   （重点看有没有 `avx512f avx512bw avx512dq avx512vl avx512_bf16`）
3. **编译器**：`g++ --version | head -1`、`clang++ --version | head -1`（有哪个写哪个）
4. **Python**：`python3 --version`；`python3 -m venv /tmp/v && echo venv-ok` 能不能建 venv？
5. **torch**：`python3 -c "import torch; print(torch.__version__)"`；有 transformers 吗？
6. **网络**：`pip download --no-deps bitsandbytes-cpu-fork==0.50.2.dev4 -d /tmp/x` 能不能通？能访问 hf-mirror.com 吗？
7. **权限**：有 sudo/root 吗？
8. **机器**：`nproc`、`free -h`、`df -h /tmp`。

## 我们拿这份报告做什么

- 确认 EPYC 的 AVX-512/BF16 标志和内核暴露情况；
- 在真实 AVX-512 机器上验证 `libbitsandbytes_cpu_avx512.so` 的加载选择和 BF16 指令安全性（这是我们发布前修的一个高危点：加载条件必须包含 `avx512_bf16`）；
- 决定下一版 wheel 的 ISA 覆盖范围（AVX2 / AVX-512 / generic / aarch64）。

有任何报错直接把原始输出贴回来即可，不用处理。
