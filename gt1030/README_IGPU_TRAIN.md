# iGPU 训练套件（任意 HuggingFace 模型）

`gt1030/igpu_train.py` 用核显（OpenCL / CLBlast）执行模型里的 Linear GEMM，CPU 只做其余算子与优化器。
它**不绑定架构**：只替换 torch 模块（`nn.Linear` 与 GPT2/NeoX 用的 `Conv1D`），所以
`AutoModelForCausalLM` / `AutoModelForMaskedLM` 能加载的模型都能训。

## 快速开始

```bash
# 本地模型目录（LoRA，合成 batch 过拟合，10 步）
python gt1030/igpu_train.py --model D:\deepsleep\_test\models\deepseek-coder-1.3b-base --steps 10

# 全参训练一个小模型（BERT masked-LM 自动识别）
python gt1030/igpu_train.py --model D:\deepsleep\tiny_models\hf-internal-testing__tiny-random-BertForMaskedLM ^
    --mode full --steps 20

# Hub 模型经 hf-mirror.com 自动下载（本地不通时用镜像）
set HF_ENDPOINT=https://hf-mirror.com
python gt1030/igpu_train.py --model Qwen/Qwen2-0.5B --mode lora --steps 50

# 真实文本 + 保存 LoRA
python gt1030/igpu_train.py --model <dir> --data corpus.txt --tokens 128 --steps 200 --save out\my_lora
```

## 参数

| 参数 | 含义 |
|---|---|
| `--model` | 本地目录或 HF repo id（自动经 `HF_ENDPOINT`，默认 hf-mirror 下载） |
| `--mode lora\|full` | 冻结 base + LoRA（默认，省内存）或全参训练 |
| `--task auto\|clm\|mlm` | 自动判断因果语言模型 / masked-LM；一般不用手写 |
| `--steps` `--tokens` `--batch` `--lr` `--lora-r` | 训练规模 |
| `--data file.txt` | 用 tokenizer 切窗训练真实文本；默认合成 batch |
| `--save dir` | 保存可训练权重（LoRA 只有 adapter）与 `train_info.json` |
| `--log-every` | 日志间隔 |

## 自动正确性对拍（重要）

每次训练开始前，脚本会：
1. 用**原始模型**在同一 batch 上算 CPU 参考 loss；
2. 替换模块后算第一步的核显 loss；
3. 打印 `parity CPU x vs iGPU y -> |diff|`，并对不上就判失败（`RESULT ... parity=`）。

这能抓住"接线错了但 loss 还在动"的静默错误。已验证的矩阵（34+ 组，详见报告 §7.123/§7.124）：

- 因果 LM：llama / mistral / qwen2 / gemma / phi3 / phi / gpt2 / gptneox / falcon / bloom /
  mixtral（MoE）/ olmo / mpt / opt / stablelm / gptbigcode / gptneo
- masked-LM：BERT / RoBERTa / DeBERTaV2 / Electra / DistilBERT
- tiny 模型全部 parity = 0.0；真实 deepseek-coder-1.3B parity = 1.9e-6

## 速度预期（诚实）

逐算子卸载的瓶颈是**每次调用的同步**（Windows WDDM + 共享内存带宽），不是算力：

| 场景 | CPU（6 线程） | 核显路径 |
|---|---|---|
| deepseek-1.3B，seq=64，LoRA | ~4.2 s/步 | **~6.9 s/步**（其中核显 GEMM 3.2–5.3 s） |
| 0.1M 参数 tiny 模型 | 0.02-0.05 s/步 | 0.03-0.09 s/步（延迟主导） |

即：**能训、能对拍、能存**，但大模型上比 CPU 慢约 1.8×。真正要更快需要设备常驻引擎
（`gt1030/llama_resident.py`，单层前向/反向已对拍通过、整模型仍在修），不在当前用户路径里。

## 内存行为

LoRA 模式下，冻结的 base 权重在**上传核显一次后即释放 CPU 副本**（1.3B fp32：设备 5.2 GB，CPU 侧
只剩 embedding/norm/LM head 等 <1 GB），因此 15.4 GB 机器可训 1.3B LoRA；全参模式保留 CPU 权重
（优化器需要），仅适合小模型。

## 已知限制

- 只支持能放进内存的模型（15.4 GB 机器上 fp32 约 ≤2B 全参；更大请用 LoRA）；
- 逐算子卸载对 **token 数小**时延迟占比高，收益在 batch 更大时才会体现；
- 绑定 `nn.Linear`/`Conv1D`：卷积模型（VAE/UNet）不在此路径；
- `lm_head` 在 tied embeddings 时保留在 CPU（避免破坏权重共享）。

## 相关文件

- `gt1030/model_matrix.py`：跨架构健康矩阵（自动跑全参/LoRA 各 6 步，JSON 输出）
- `gt1030/igpu_train_lora.py`：早期单模型原型（保留做对比）
- 报告：`D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md` §7.121–§7.124
