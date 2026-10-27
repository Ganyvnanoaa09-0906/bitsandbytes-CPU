"""Add section 7.111: the overfit-100 run (20 epochs) and what it actually proves.

Result: teacher-forced reconstruction of training rows is recognisable illustration content, but
autoregressive roll-out collapses even from an 896-token true prefix, and this run used the old
no-BOS path (generate starts from token ID 0 -- the code names this the root cause of cold-start
collapse). So the §7.50.5 dichotomy resolves to "the model learns the images under teacher forcing;
the failure is in the decode", with a named confound that has to be removed by rerunning with --use_bos.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PATH = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW = '''### 7.111 ★★★ 过拟合 100 张的 20 个 epoch：教师强制能重建出可辨认插画，而自由采样仍是纹理

#### 7.111.1 这是 10-15 留下的那个崩溃实验的复跑

**§7.50.5 设计的判据是"模型能不能拟合这 100 张"：**

| 结果 | 结论 |
|---|---|
| 采样出那 100 张的可辨认版本 | 架构没问题，问题是算力 |
| 仍然只是纹理/色块 | 架构或采样流程有结构性问题 |

**那次在 10-15 23:22 启动，第 1 步之后 `train exit -1`，之后一直没复跑。
今晚先做 5 步 smoke test（exit 0）再整跑：200 行（100 张 ×2）· T=1024 · batch 2 ·
30.50M · Muon(NS=3, wd 0.05) + AdamW8bit(wd 0.05) · lr 1e-4 · 2000 步 = 20 epoch。**

```
2000 步 × 4.8 s ≈ 2.72 小时（19:12 → 21:57，分 6 段采样：100/250/500/1000/1500/2000）
峰值 RSS 1.93 GB · 训练 loss 从 7.30 到 2.02
```

#### 7.111.2 曲线：train 在降，val 在涨（过拟合开始）

| 步 | train | val | gap |
|---|---|---|---|
| 1 | 7.296 | — | — |
| 61 | 6.941 | 6.927 | −0.014 |
| 221 | 4.295 | 6.438 | +2.14 |
| 981 | 3.475 | 7.538 | +4.06 |
| 1481 | 2.329 | 8.020 | +5.69 |
| **1981** | **2.016** | **8.213** | **+6.20** |

**★ 但"过拟合"不等于"背下来"：20 epoch 后 train CE 仍约 2.0 nats（随机 6.93），
离逐位复现（CE→~0）还差得远。且 val 只有固定 4 行（2% 划分），波动大，别过度解读。**

#### 7.111.3 自由采样：六个检查点【全是纹理】

**step 100/250/500/1000/1500/2000 的 8 张采样全部是纹理/色块，没有一张是那 100 张的可辨认版本。
（样本：`i5build/ar_overfit100_v3/samples_step*.png`）**

#### 7.111.4 ★★★ 但教师强制重构是【可辨认的插画】—— 这才是今晚的发现

**用同一份最终 ckpt，对训练行做"首 token 真值 + 其余位置 argmax"的一次前向重构：**

```
移位 CE（训练同口径：logits[:,i] 预测 tokens[:,i+1]）  = 2.394 nats（随机 6.931）
argmax 逐位一致率                                   = 46.2%
  前 1/4 位置 52.8% · 后 1/4 位置 43.2%
```

**而把重构的 token 解码成图（`TF_GREEDY.png`）—— 是可辨认的插画：主体、构图、背景都在，
只是细节糊。对照真值（`TRUE.png`）两图几乎一一对应。**

**⇒ ★【所以模型【不是】只学了局部纹理：在教师强制下它学到了这 100 张的构图与主体。
§7.50.5 判据里的"架构不能拟合"这一支，被这条否掉了。】**

#### 7.111.5 ★★★ 而自回归 rollout 会崩 —— 独立于 BOS 的第二个病根

**给 896 个真 token（整图 87.5%），只让它续 128 个：**

```
第 1 个续 token：真值 rank 7（prob 0.037），argmax 给了另一个码
greedy 续 128 个    ⇒ 只剩 6 个不同码（塌成少数 token），一致率 0%
top-k 100 采样续    ⇒ 一致率 0/128（随机基线 0.098%）
从 512 前缀续       ⇒ "36.7% 一致率"，但真值那段本身就是同一个背景 token 803 重复
                       ⇒ ★【不是记住，是猜中了众数】
```

**⇒ ★【一个错 token 就把上下文带离训练流形，之后越走越偏 —— 这是 exposure bias，
与"有没有 BOS"无关（896 个真 token 已经排除了冷启动的影响）。】**

#### 7.111.6 ★★ 一个必须点名的混淆：本次跑的是【没有 BOS】的旧路径

**代码里写得很清楚（`small_image_model_v2.py` 的 generate）：**

```
use_bos=False：旧行为，从 token ID 0 起步（这就是冷启动崩坏的根源）
```

**本次 run 用默认 `--use_bos False`（日志确认），所以【自由采样】【从 token 0 冷启动】——
这是已知的 BOS bug，不是新证据。而 7.111.5 的 896-token 前缀实验【不受】这个混淆影响
（前缀已经给了 87.5% 的真序列）。**

**⇒ ★【所以今晚能下的结论只有两条：**
1. **教师强制下：模型学到了这 100 张的构图与主体（重构可辨认）** —— 架构能拟合；
2. **自回归解码下：rollout 从第一个错 token 起就崩** —— 病在解码，不在"学不会"。
**而"自由采样是否只差 BOS"必须用 `--use_bos` 重跑同一套 20 epoch 才能判**（本次没跑 BOS 路径）。

#### 7.111.7 下一步（有界，按性价比排序）

| # | 实验 | 代价 | 回答什么 |
|---|---|---|---|
| ① | 同一 100 张 + `--use_bos`，2000 步 | 2.7 h | 自由采样崩坏里有多少是 BOS，多少是 exposure bias |
| ② | 单图（2 行）过拟合到 CE→0 | ~20 min | 模型对【单张】的容量上限；本次因进程管理失败没跑成 |
| ③ | 896 前缀 + 教师强制微调（scheduled sampling / GRN 已有开关） | 2-3 h | 能否把 rollout 的误差累积压下去 |

**★ 而"加数据/加算力"不在这个列表里 —— 因为 7.111.4 已经说明模型能拟合，
7.111.5 说明崩在解码；这两条把问题从"数据量"移到了"解码"。**

#### 7.111.8 复现

```
overfit100_v3.py        → overfit100_v3.log · ofv3_seg_*.log · i5build/ar_overfit100_v3/
overfit100_verdict.py   → evidence/raw_outputs/overfit100_verdict_20261017.txt
                          evidence/overfit100_verdict/{TRUE,TF_GREEDY}.png
overfit100_cont_diag.py → evidence/raw_outputs/overfit100_cont_diag_20261017.txt
probe_exposure.py       → evidence/raw_outputs/overfit100_exposure_20261017.txt
```



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW + '# 附录 A：AR 训练路径的实测证据', 'insert 7.111'),
    ('**本文是 110 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     '**本文是 111 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     'header: 111'),
    ('**本报告的 110 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     '**本报告的 111 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     'F header: 111'),
    ('| MaskGIT | ⚠️ **实现完整而从未被执行过** —— 第一次真跑是 10-15 深夜 | §7.47 | ⚠️ |',
     '| MaskGIT | ⚠️ **实现完整而从未被执行过** —— 第一次真跑是 10-15 深夜 | §7.47 | ⚠️ |\n'
     '| **过拟合 100 张（20 epoch）** | ★ **教师强制重构可辨认（模型能拟合）；自由采样仍纹理；896-token 前缀 rollout 也崩 ⇒ 病在解码** | §7.111 | ★★★ |',
     'E table: overfit row'),
]


def main(dry):
    text = open(PATH, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        n = out.count(old)
        if n != 1:
            print('ABORT %-20s occurrences=%d' % (label, n))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    print()
    print('chars %d -> %d · lines %d -> %d' % (len(text), len(out),
                                               text.count('\n') + 1, out.count('\n') + 1))
    if dry:
        print('(dry run)')
        return 0
    with open(PATH, 'w', encoding='utf-8', newline='') as f:
        f.write(out)
    print('written')
    return 0


if __name__ == '__main__':
    sys.exit(main('--dry' in sys.argv))
