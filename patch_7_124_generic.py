"""Report 7.124: universal iGPU training kit v2 -- automatic CPU/iGPU loss parity, 34/34 matrix.

Also fixes the param-count double count (base_weight was added on top of the pre-patch parameter count).
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW = '''### 7.124 通用 iGPU 训练套件 v2：每次训练自动做 CPU/iGPU 对拍，34/34 通过

§7.123 的套件已经跨架构可用，但"能跑"不等于"训得对"。这一节加了**每次训练自动对拍**：在替换任何
模块之前，先用原始模型在同一 batch 上算一遍 CPU 参考 loss；替换后第一步的 iGPU loss 必须与它一致，
否则直接判失败。这能抓住"接线错了但 loss 还在动"的静默错误——报告里反复记过的那类失败。

**实现**（`gt1030/igpu_train.py`）：

- 架构无关：只替换 torch 模块（`nn.Linear` 与 HF `Conv1D`），`AutoModelForCausalLM` 能加载的
  任何因果语言模型都可训；本地目录或经 `hf-mirror.com` 自动下载；
- 两种模式：`--mode lora`（冻结 base + LoRA，AdamW 只维护 adapter 状态）与 `--mode full`（全参）；
- 真实数据：`--data file.txt` 走 tokenizer 切窗；`--tokens/--batch` 控制序列；
- 保存：`--save dir` 保存可训练权重 + JSON 训练信息；
- **对拍**：输出 `parity CPU x vs iGPU y -> |diff|`，并写进 RESULT 行（`parity=`）。

**跨架构矩阵**（17 架构 × {全参, LoRA} = 34 次运行；tiny 模型，6 步）：

| 项 | 结果 |
|---|---|
| 运行数 | 34 |
| 有限且 iGPU 调用 > 0 | **34/34** |
| **CPU/iGPU loss parity** | **全部 0.0**（fp32 Sgemm 逐位一致） |
| 步时间 | 0.02–0.09 s/step（tiny 模型；延迟主导） |

覆盖：llama / mistral / qwen2 / gemma / phi3 / phi / gpt2（Conv1D 路径）/ gptneox / falcon /
bloom / mixtral（MoE）/ olmo / mpt / opt / stablelm / gptbigcode / gptneo。唯一排除的
`ydshieh/tiny-random-gptj` 是 checkpoint 自身损坏（rotary_dim=64 > head_dim=8，原模型同样报错）。

**真实模型**：deepseek-coder-1.3B（LlamaForCausalLM，169 个 Linear，LoRA 7.77 M 可训）：
parity **1.91e-6**，**8.2 s/步**（seq=64；iGPU 调用 3.8 s），loss 11.72 → 8.89（3 步）。
另起了一个 500 步的后台训练，用于验证长时间稳定性与保存权重。

**通用性相关的三个 bug（都在这一轮修掉）**：

1. **in-place 视图错误**：Falcon/Olmo 会原地 `+=`/`clamp_` qkv 切片，而 autograd Function 返回的
   reshape 视图禁止原地修改 → 输出 numpy 数组直接按最终形状分配，Function 不再返回视图；
2. **全参训练没有权重梯度**：weight 现在参与 autograd（iGPU 上用转置 GEMM 算 dW），不再只有 dX；
3. **GPT2 的 Conv1D 布局**：权重是 (in,out)，Function 带 layout 标记，前向 `x @ W`、dX/dW 用对应
   的转置组合。

**⇒ 判决：任意 HF 因果语言模型"能训、能对拍、能存"这一层已经打通（通用性交付）。** 速度上逐算子
混合仍比 CPU 慢 ~1.8×（1.3B/seq64：8.2 s vs 4.2 s），这是 §7.121 已归因的同步税；设备常驻引擎
（§7.122）才是速度交付，但它目前卡在一个**多层前向的 GPU 缓冲区异常**上（单层前向/反向已对拍通过；
多层时激活缓冲区读出全 0 或互相相同，且在不同进程间不稳定，怀疑与驱动/上下文生命周期有关），
已明确标为未完成、不能进用户路径。

'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW + '# 附录 A：AR 训练路径的实测证据', 'insert 7.124'),
    ('**本文是 123 节的【时序记录】', '**本文是 124 节的【时序记录】', 'header: 124'),
    ('**本报告的 123 节里有 12 节', '**本报告的 124 节里有 12 节', 'F header: 124'),
    ('| **真实模型核显训练（deepseek-1.3B LoRA）** | ★ **精度等价（fp32 Sgemm，loss diff ≤0.02）；核显利用率 87–124%，但逐算子同步使速度比 CPU 慢 2–3×** | §7.121 · `gt1030/igpu_train_lora.py` |',
     '| **真实模型核显训练（deepseek-1.3B LoRA）** | ★ **精度等价（fp32 Sgemm，loss diff ≤0.02）；核显利用率 87–124%，但逐算子同步使速度比 CPU 慢 2–3×** | §7.121 · `gt1030/igpu_train_lora.py` |\n'
     '| **通用 iGPU 训练套件（任意模型）** | ★ **17 架构 ×2 模式 34/34 通过，CPU/iGPU parity 全 0.0；真实 1.3B parity 1.9e-6；--data/--save/自动下载** | §7.124 · `gt1030/igpu_train.py` |',
     'H table: 7.124 row'),
]


def main() -> int:
    text = io.open(REPORT, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        if out.count(old) != 1:
            print('ABORT %-20s anchor=%d' % (label, out.count(old)))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('report chars %d -> %d, lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
