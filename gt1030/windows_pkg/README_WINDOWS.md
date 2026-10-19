# CPU-forge 核显训练包（Windows）

用 **核显（OpenCL / CLBlast）** 训练 HuggingFace 模型：模型里的 `nn.Linear`（以及 GPT2/NeoX 的
`Conv1D`）交给核显做 GEMM，CPU 负责其余算子和优化器。任何 `AutoModelForCausalLM` /
`AutoModelForMaskedLM` 能加载的模型都可以——不绑定具体架构。

> 诚实的预期：逐算子卸载的瓶颈是**每次调用的同步**（Windows WDDM + 共享内存带宽），不是浮点
> 算力。大模型上比纯 CPU 慢约 **1.8×**（deepseek-1.3B seq=64 LoRA：CPU ~4.2 s/步，
> 核显 ~6.9 s/步）。它的价值是"能训、能对拍、能存"，以及在 CPU 被其它工作占满时把 GEMM
> 挪走。想更快需要设备常驻引擎，仍在开发中，不在本包路径里。

## 1. 前置条件

- Windows 10 / 11，64 位；
- **Python 3.10 – 3.12 64 位**（python.org 安装时勾选 *Add python.exe to PATH*）；
- 显卡驱动：AMD（Adrenalin，自带 OpenCL）、Intel（Arc/Iris 驱动）、NVIDIA（驱动自带 OpenCL）；
- 磁盘 ~2 GB（venv + torch CPU wheel）；内存建议 ≥ 8 GB（LoRA 1.3B 需要 ~6 GB 可用）。

## 2. 三步开始

```bat
setup.bat            :: 第一次：建 .venv、装 torch/numpy/pyopencl/transformers，然后自检
doctor.bat --bench   :: 随时检查：torch、CLBlast、OpenCL 设备、iGPU 拷贝带宽
train.bat --model sshleifer/tiny-gpt2 --steps 5          :: 最小冒烟测试
```

国内网络如果 torch 下载慢：`setup.bat mirror`（走清华 PyPI 镜像）。

## 3. 常用训练命令

```bat
:: 下载一个小模型跑 LoRA（默认经 hf-mirror.com）
train.bat --model sshleifer/tiny-gpt2 --steps 20

:: 本地模型目录 + LoRA
train.bat --model D:\models\qwen3.5-0.8B --steps 50

:: 真实文本文件，切窗大小 128，跑 200 步并保存 adapter
train.bat --model D:\models\qwen3.5-0.8B --data corpus.txt --tokens 128 --steps 200 --save out\my_lora

:: 全参训练（只建议小模型；大模型会 OOM）
train.bat --model D:\models\tiny-bert --mode full --steps 20

:: 没有可用 OpenCL 设备时，退回纯 CPU 基线
train.bat --backend cpu --model D:\models\x --steps 20

:: 只做 CPU/iGPU 对拍检查（2 步，打印 parity）
train.bat --model D:\models\x --check
```

每次训练开始前脚本会自动做 **CPU/iGPU 对拍**：用原始模型算 CPU 参考 loss，替换模块后算第一步
核显 loss，打印 `parity |diff|`；对不上会判失败。这是为了抓"接线错了但 loss 还在动"的静默错误。

## 4. 支持的模型（已对拍矩阵 44/44）

- 因果 LM：llama / mistral / qwen2 / qwen3 / gemma / phi3 / phi / gpt2 / gptneox /
  falcon / bloom / mixtral（MoE）/ olmo / mpt / opt / stablelm / gptbigcode / gptneo
- masked-LM：BERT / RoBERTa / DeBERTaV2 / Electra / DistilBERT
- 真实模型验证：deepseek-coder-1.3B parity = 1.9e-6（fp32）

## 5. 内存与保存

- **LoRA（默认）**：冻结的 base 权重上传核显一次后释放 CPU 副本（1.3B fp32：设备 5.2 GB，
  CPU 侧只剩 embedding/norm/LM head <1 GB），15.4 GB 机器可训 1.3B LoRA；
- **全参**：CPU 侧保留权重和优化器状态，只建议小模型；
- `--save dir` 保存 `trainable.pt`（LoRA 只有 adapter）+ `train_info.json`；
- `--resume dir` 从 `trainable.pt` 恢复继续训练。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `OpenCL        : no platforms` | 更新显卡驱动（AMD Adrenalin / Intel 驱动 / NVIDIA 驱动）；仍无则用 `--backend cpu` |
| `CLBlast dll   : NOT FOUND` | 确认 `clblast.dll` 与脚本同目录；或设 `CLBLAST_PATH=完整路径` |
| `torch         : MISSING` | 先跑 `setup.bat`，或手动 `pip install torch`；国内用 `setup.bat mirror` |
| Python 版本不对 | 安装 3.10–3.12 64 位；`where py` 检查 |
| 模型下载慢/失败 | 脚本默认走 `hf-mirror.com`；也可先手动下载模型目录，用本地路径 |
| 内存不足 | 用 LoRA、减小 `--tokens/--batch`、换更小模型；全参模式尤其吃内存 |
| `parity` 对不上 | 把完整输出发回（模型名/架构），这是需要修的接线 bug |
| 训练很慢 | 先跑 `doctor.bat --bench` 看 iGPU 拷贝带宽；逐算子卸载在 token 少时延迟占比高，加大 batch 才有收益 |

## 7. 文件清单

```
setup.bat            一次性安装（可选 mirror）
doctor.bat           环境检查（--bench 测 iGPU 带宽）
train.bat            训练入口（参数透传给 igpu_train.py）
check_env.py         环境检查实现
igpu_train.py        训练器
clblast.dll          CLBlast（Apache-2.0，见 THIRD_PARTY_NOTICES.md）
README_WINDOWS.md    本说明
THIRD_PARTY_NOTICES.md
```

## 8. 第三方许可

CLBlast：Apache License 2.0（版权见 `CLBlast-LICENSE.txt`）。torch（BSD）、transformers（Apache-2.0）、
pyopencl（MIT）等由 pip 安装，许可随各自包分发。详见 `THIRD_PARTY_NOTICES.md`。
