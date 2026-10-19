# CPU-forge 全项目代码审查报告

日期：2026-10-19（0.50.2.dev4 发布前）
审查者：自动审计 + 人工重点审查
范围：`D:\work\bitsandbytes-CPU` 全仓库（1952 个 git 跟踪文件；5820 个文件含模型缓存/日志等未跟踪产物）

## 1. 审查方法

| 层次 | 方法 | 结果 |
|---|---|---|
| 全仓库语法 | 947 个 `.py` 全部 `compile()`；21 个 `.sh` 全部 `bash -n` | 全部通过 |
| 静态模式扫描 | 凭据（pypi-/sk-/AKIA/私钥）、`eval/exec/os.system/shell=True/pickle.load/torch.load/verify=False`、裸 `except`、静默 `except: pass`、`-march=native`、硬编码绝对路径 | 无任何凭据；shipped 包内 0 危险调用；详见 §4 |
| C/C++ 编译 | x86_64 Linux（AVX2/AVX-512/generic）、aarch64-linux（Termux 路径）、x86_64-windows-gnu（clang，两种 ISA）；`-Wall -Wextra` | 全部通过；仅 141 条 unused-variable 警告（无功能影响） |
| 发布件验证 | 4 个产物 `twine check`；Windows wheel 干净 venv 安装 + import + C 自检；两个 Linux wheel 在 WSL（torch 2.9.1+cpu）安装 + import + C 自检 + CLI | 全部 PASS |
| 重点人工审查 | `csrc/*`（本次改动）、`cextension/_cpu_caps/gdn_cpu/disk_balancer/torch_cpu_kit/detect_cpu/efst/cli`、打包配置、`gt1030/igpu_train.py`、`llama_resident.py` | 发现并修复下列问题 |
| 端到端验证 | Qwen3.5-0.8B 真实模型 GDN patch 前/后 hidden state 对比；EFST 3 组回归；disk_balancer 安全回归；`igpu_train --backend cpu` 2 步冒烟 | 全部 PASS（GDN max\|d\| 2.0e-3 / \|x\| 41） |

## 2. 发现并修复的问题

### 高危（发布阻断级）

1. **AVX-512 库含 BF16 指令，但加载条件没查 BF16 → 部分 Intel CPU 会 SIGILL**
   - `build_linux.sh` 在编译器支持时加 `-mavx512bf16`；实测 `libbitsandbytes_cpu_avx512.so` 里有 **182 条** `vcvtneps2bf16/vcvtne2ps2bf16`（auto-vectorise 到普通 kernel）。
   - `_cpu_caps.avx512_supported()` 只要求 `avx512f/bw/dq/vl`，Skylake-X/Cascade Lake 满足但无 BF16 → 载入即可能非法指令。
   - 修复：要求 `avx512_bf16` flag；Linux 以 `/proc/cpuinfo` 为准，只有 torch 的 "AVX512" 能力时保守返回 False（可用 `BNB_CPU_FORCE_AVX512=1` 强制）。
   - 回归测试：`test_cpu_caps_selection.py`。

2. **disk_balancer `cleanup()` 会整目录 `rmtree` 用户给的 `--flash_paths` → 数据丢失风险**
   - 用户路径若指向已有目录（甚至盘根），默认 `keep_cache=False` 下 cleanup 会删光。
   - 修复：用户路径下固定使用 `.bnb_flash_cache/` 子目录；cleanup 只删本实例创建并打标记的目录。
   - 回归测试：`test_disk_balancer_safety.py::test_cleanup_preserves_user_directory`。

3. **新增 CPUID 头在 Windows/clang 目标上编译失败**
   - clang `<cpuid.h>` 把 `__cpuid` 定义成函数式宏，Windows 头 `psdk_inc/intrin-impl.h` 又声明 `void __cpuid(int[4], int)` → 宏展开报错；这会让"Windows 无 MSVC 交叉构建"路径直接断掉（MSVC 路径不受影响，但仍须保证 clang/MinGW 可编译）。
   - 修复：include 后 `#undef __cpuid`（保留 `__cpuid_count`，`cpu_cache.h` 在用）；x86_64-windows-gnu 的 generic/AVX2 编译恢复；aarch64 不受影响。
   - 注：仓库里分发的 Windows DLL 是历史 MSVC 构建（imports VCOMP140），本次无法在本机重建；其代码路径不经过新头，行为不变。

4. **EFST 三个真实 bug**（用户反馈"efst.py 有 bug"）
   - 嵌套 MoE 块（block 同时有 `gate` 与 `experts`）被识别成 2 个"专家"→ 路由器被当专家、非选中专家永远冻结不了；修复为三层优先级（3D 张量专家 → ModuleList/ModuleDict → 兜底）并丢弃祖先候选。
   - 3D 张量专家 forward 用 `F.linear` 语义，但 Qwen3Next/Qwen3.5 权重是 `[E, in, out]`（grouped-mm 语义）→ 形状不匹配；改为按最后一维自动判别两种布局。
   - 非 MoE 模型上 `apply_efst` 只告警然后冻结全部参数（可训练数 0，训练静默无效）；现在默认 `ValueError`，需要时显式 `EFSTConfig(allow_no_experts=True)`。
   - 回归测试：`test_efst_regressions.py`（嵌套块 / 3D 张量专家 / 非 MoE 安全），`test_efst_full.py` 5/5，`efst_selftest.py` PASS，`verify_efst_8bit.py` PASS。

### 中危

5. **disk_balancer 磁盘配额从未生效**：`max_cache_bytes`/`--flash_sizes` 只计算不检查，缓存可写满磁盘。修复：按盘记账，超额时拒绝本次卸载（参数保留内存）并告警。
6. **disk_balancer `get_cold()` 会顺手 `_latent_store.cleanup()`**：读一个冷参数会清空无关的 latent 分块缓存。已移除。
7. **cextension 静默降级**：native 库加载失败时只装 mock，之后每个 kernel 调用才报错。修复：import 时先 `logger.warning` 指路 `python -m bitsandbytes`。
8. **pre-AVX2 x86 上会载入 AVX2 库导致 SIGILL**（Windows wheel 只带 AVX2 DLL）：现在若缺 generic 库且是 x86_64，直接抛可读错误而不是载入不兼容代码。
9. **`gt1030/igpu_train.py`**：
   - 整个训练循环停在 `model.eval()`（dropout 被静默关闭）；已改为参考 loss 用 eval，训练前 `model.train()`。
   - 优化器参数列表未去重：tied embeddings / tied lm_head 在 full 模式可能同一张量进两次，AdamW 一步更新两次；已按 `id()` 去重。
   - Hub 下载静默失败（只建了半个目录，后面报 "Unrecognized model"）；现在校验 `config.json` + 权重并抛出明确错误；Hub 根目录支持 `BNB_HUB_CACHE`。
   - `--data` 缺 tokenizer 时是 AttributeError；现在是明确 RuntimeError。
   - `torch.load` 加 `weights_only=True`（带旧 torch 回退）。
10. **`detect_cpu.py` winreg key 未关闭**（资源泄漏）；已用 `with`。
11. **`llama_resident.py` 文档不诚实**：full-model 多层的 GPU buffer 异常是 WIP，原 docstring 没写；已补充 STATUS 段并说明 import 时创建 OpenCL 上下文的行为。

### 低危 / 环境提示

- csrc 有 141 条 unused-variable 警告（AVX-512/generic 条件编译分支），无功能影响，暂不清理以免动发布二进制。
- 项目脚本里存在 201 处 `except: pass`-类静默分支、9 处硬编码盘符路径、132 处未指定 `weights_only` 的 `torch.load`——均不在 PyPI 包内（shipped 包为 0）。
- `-march=native` 只出现在历史 patch 生成脚本中，不参与构建。
- 本机系统 Python 同时存在：site-packages 里旧的 dev0 安装 + editable `.pth` 指向源码 → 直接 `python xxx.py` 可能 import 到旧包。发布验证已全部在干净 venv / WSL 中进行；仓库测试脚本已确保把源码目录插到 `sys.path` 最前。
- 全仓库无任何 PyPI/OpenAI/AWS token 或私钥。

## 3. 验证结果（发布件）

| 产物 | 安装 | import | C 自检 | 备注 |
|---|---|---|---|---|
| `...-py3-none-win_amd64.whl` (0.44 MB) | 干净 venv PASS | PASS | PASS | avx2=True, avx512=False（保守，Windows 未带 avx512 DLL） |
| `...manylinux2014_x86_64.manylinux_2_17_x86_64.whl` (2.76 MB) | WSL PASS | PASS | PASS | 可移植 serial 版（zig 交叉编译，GLIBC≤2.16） |
| `...manylinux_2_39_x86_64.whl` (2.15 MB) | WSL PASS | PASS | PASS | 本机 OpenMP 版（libgomp 由 auditwheel 打包） |
| `...tar.gz` (0.80 MB) | — | — | — | 含 csrc/cpu_features.h/build_linux.sh，Linux 源码安装可自编 |

`twine check` 4/4 PASSED。

## 4. 遗留风险与开放项

1. **PyPI 上传**：产物已就绪，但本机没有 API token（`tools/publish.ps1` 设计为隐藏输入）。需要用户提供 token 或自行执行 `publish.ps1`。
2. **GitHub push** 仍被网络重置阻断（本地领先 6+ commits），恢复后需补推。
3. **Windows DLL 未随本次 csrc 改动重建**（本机无 MSVC；用 zig 重建会改变依赖/导出，故保留已验证的 MSVC 产物）。MSVC 分支不经过新 CPUID 头，功能等价。
4. **AVX-512 Windows DLL 不发**：Windows 用户固定 AVX2；这是刻意保守选择。
5. `llama_resident.py` 多层路径的 GPU buffer bug 仍未修（已标 WIP，非发布路径）。
6. 上游 CUDA/GPU 代码路径未深审（本机无 GPU，且不在 fork 改动范围）。

## 5. 本次审查产生的代码/测试文件

- 修复：`bitsandbytes/bitsandbytes/_cpu_caps.py`、`cextension.py`、`disk_balancer.py`、`detect_cpu.py`、`efst.py`（含包内 `bitsandbytes/efst.py`）、`bitsandbytes/csrc/cpu_features.h`、`bitsandbytes/setup.py`、`bitsandbytes/pyproject.toml`、`gt1030/igpu_train.py`、`gt1030/llama_resident.py`、`efst.py`（根目录兼容 shim）。
- 新增测试：`test_efst_regressions.py`、`test_cpu_caps_selection.py`、`test_disk_balancer_safety.py`、`verify_gdn_patch_real_model.py`、`audit_project.py`。
- 文档：`EFST_README.md` 修复记录；本文件；技术报告 §7.126。
