# EFST — Expert-Specific Fine-Tuning（专家专项微调）

EFST 用于 MoE 模型在 **纯 CPU / 低内存** 环境下的微调：

- 只微调被选中的专家（以及可选的 router），冻结其余全部参数；
- 冻结参数不产生梯度，也就不占用优化器状态，直接减少内存；
- 可给选中专家内部注入 LoRA，进一步把可训练参数量压到 adapter 级别；
- 不依赖具体模型实现，自动识别常见 `experts` / `moe` / `block_sparse_moe` 结构。

## 快速开始

```python
import torch
from efst import EFSTConfig, apply_efst

model = AutoModelForCausalLM.from_pretrained("some-moe-model")
model.train()

config = EFSTConfig(
    # 方式 A：手动指定专家
    # expert_indices={"mlp.experts": [0, 3, 7]},
    # 方式 B：用校准数据自动选最热的 2 个专家
    top_k=2,
    calibration_dataloader=train_loader,
    calibration_forward_fn=lambda batch: model(**batch),
    num_calibration_batches=8,
    tune_router=True,      # 是否同时微调 router/gate
    lora=True,             # 专家内 LoRA
    lora_r=8,
    lora_alpha=16,
)

info = apply_efst(model, config)
print(info.summary())
```

之后正常用 `Trainer` 或手写训练循环即可。

## 不想要 LoRA，直接全专家微调？

```python
config = EFSTConfig(
    top_k=2,
    calibration_dataloader=train_loader,
    calibration_forward_fn=lambda batch: model(**batch),
    lora=False,
)
```

## API

| 函数 | 作用 |
|---|---|
| `find_expert_groups(model)` | 自动找出 MoE 专家容器（含 3D 张量专家） |
| `freeze_all(model)` | 冻结全部参数 |
| `unfreeze_experts(model, expert_indices, tune_router)` | 只解冻指定专家 |
| `collect_expert_usage(model, dataloader, ...)` | 统计路由调用次数 |
| `select_top_experts(model, dataloader, top_k, ...)` | 按路由热度自动选专家 |
| `add_lora_to_experts(model, expert_indices, ...)` | 给选中专家注入 LoRA |
| `apply_efst(model, config)` | 一键应用 EFST |
| `report_trainable(model)` | 打印可训练参数分布 |
| `split_3d_expert_params(module)` | 把 3D 张量专家权重拆成 ParameterList（按专家行解冻的前提） |

## 3D 张量专家支持（transformers 5.15+ Qwen3Next/Qwen3.5）

新版 transformers 的 MoE 专家权重是 ``[num_experts, ...]`` 的大 Parameter
（``Qwen3NextExperts`` 的 ``gate_up_proj``/``down_proj``），没有独立子模块。
EFST 的处理：

- **识别**：类型名含 expert/moe 且持有 3D 参数 → 建 tensor 专家组；
  容器（``mlp`` 块）不再误当专家容器；
- **路由统计**：forward 的 ``top_k_index`` 参数逐 token 计数，按专家粒度选热专家；
- **按行解冻**：``split_3d_expert_params`` 把 3D 张量拆成
  ``ParameterList``（PyTorch 的 requires_grad 是参数级，不拆没法按行冻），
  同时把 forward 换成逐专家循环版（索引语义不变）；
- **LoRA**：tensor 专家没有 nn.Linear 子模块，无法注入 LoRA —— ``lora=True``
  时对 tensor 专家自动回退为直接解冻选中专家行（经典专家仍走 LoRA）；
- **注意**：拆分后 ``state_dict`` 的键从 ``gate_up_proj`` 变为
  ``gate_up_proj.0``、``gate_up_proj.1`` ...（结构改变，与 LoRA 注入同理），
  保存/加载模型时需保持一致（先 apply_efst 再加载权重）。

实测（i5-10400, 随机初始化 Qwen3Next 3 层混合模型）：
3 个 tensor 专家组 × 8 专家，top-2 选择 → 可训练参数 725,712 → 143,600
（19.8%），60 步训练 loss 下降 23% 且收敛趋势正常。

## 和 CPU 训练包配合

推荐顺序：

```python
import torch_cpu_kit as tck
tck.apply_env()          # import torch 之前

import torch
tck.init(verbose=True)

# bitsandbytes GDN 加速
import bitsandbytes as bnb
from bitsandbytes.gdn_cpu import patch_transformers
patch_transformers()

# 加载 MoE 模型后应用 EFST
from efst import EFSTConfig, apply_efst
apply_efst(model, config)

# 8-bit 优化器只会在可训练参数上分配状态
opt = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

## 低内存提示

- EFST 冻结非目标专家后，`AdamW8bit` 只会给可训练参数分配状态；
- 如果内存仍然紧张，可以配合 `torch_cpu_kit` 的 `mem_report()` 周期性观察可用内存；
- 后续可以把“非选中专家”进一步做磁盘 offload，只在对应层 forward 前临时载入。
