"""Record the two decisive memorisation experiments in the report (new section 7.119).

Results:
  * single-image (2 rows, 300 steps): loss 7.48 -> 0.0677, shifted CE 0.089 nats, greedy
    reconstruction 97.56% -- the architecture/recipe CAN memorise; the TF image is visually
    identical to TRUE.
  * overfit-100 with --use_bos (2000 steps = 20 epochs): train loss 2.486 at step 1981 while the
    fixed val set rises 6.91 -> 8.08 (overfitting); on training rows: CE 2.131 nats, greedy 50.20%.
    Same recipe with use_bos=False (7.111 run): CE 2.767, greedy 40.28%. So BOS is real (+0.64 nats,
    +9.9pp) but does not fix free sampling -- the samples are still pale/blurred.
  * method note: eval_memorize first used the shifted convention on the BOS checkpoint, reporting a
    bogus CE 6.64 / 15.6%; with use_bos=True logits[:, i] already predicts tokens[:, i], so the
    evaluator had to switch conventions. The corrected numbers are the ones above.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW_SECTION = '''### 7.119 ★★★ 记忆判决：单图能背到 CE 0.09；BOS 真实有效但不解决自由采样

#### 7.119.1 两个实验分别回答什么

§7.111 留下的问题有两个：**（a）30.5M 的 AR 架构到底能不能拟合？**（b）**BOS 混淆**——那次
`use_bos=False`，自由采样从 token ID 0 冷启动，所以"自由采样塌陷"混着 BOS bug 与 exposure bias，
无法分离。两个实验（`overfit_single.py` → `overfit100_bos.py`，顺序跑，同一 `eval_memorize.py`
同一把尺）：

1. **单图 2 行 × 300 步**：小到没有数据借口，若仍记不住，就是架构/配方问题；
2. **100 图 × 2 行 × 2000 步 = 20 epoch，`--use_bos`**：与 §7.111 唯一差别是 BOS，其余同配置。

#### 7.119.2 单图：架构能背，而且背得很干净

| 指标 | 结果 |
|---|---|
| 训练 loss | 7.48 → **0.0677**（step 300） |
| 教师强制移位 CE | **0.0891 nats**（随机 6.931） |
| 贪心重构逐位一致率 | **97.56%** |
| 视觉 | TF 解码图与 TRUE **肉眼无差**（`i5build/of_single/memorize_{TRUE,TF}.png`） |

**⇒ 容量/优化器/内核这条链没有问题。** "记不住 100 张"不能怪架构不会拟合。

#### 7.119.3 100 图 + BOS：BOS 是真实因素，但不是自由采样的病根

| 指标 | §7.111（use_bos=False） | 本实验（use_bos=True） | 差 |
|---|---|---|---|
| 教师强制 CE（训练行） | 2.767 nats | **2.131 nats** | **−0.64** |
| 贪心重构逐位一致率 | 40.28% | **50.20%** | **+9.9pp** |
| 固定验证集 loss | —（那次 val 6.3~6.9） | 6.91 → **8.08**（step 61→1981 上升） | 过拟合 |
| 训练集 loss（step 1981） | — | 2.486 | 背训练集 |
| 自由采样视觉 | 抽象色块 | **仍糊**（苍白/大色块，`samples_step2000.png`） | 未解决 |

**⇒ ★【BOS 修复了"冷启动语义"，真的让教师强制重构更像（CE 降 0.64、一致率 +10pp）；
但自由采样依旧崩坏 ⇒ 主要病因是 exposure bias（训练只见真前缀、推理见自己的错），
叠加 196 行训练集上的过拟合（val 6.91→8.08、train 2.49）。】**

#### 7.119.4 一个方法学插曲：评估口径必须跟训练口径一致

第一次评 BOS checkpoint 时，`eval_memorize.py` 仍用旧版**移位**约定（`logits[:, i]` 对
`tokens[:, i+1]`），得到 CE **6.637 / 一致率 15.62%**——几乎等于随机，与训练日志的 train 2.49
矛盾。根因：`use_bos=True` 时 BOS 已提供 shift，`logits[:, i]` **直接**预测 `tokens[:, i]`
（`forward_logits` 的 docstring 写得很清楚）。改成 BOS 约定后：**2.131 / 50.20%**。

**⇒ 这是本报告第 N 次"口径与语义不一致"的失效**（同 §7.71 的内存延迟、§7.118 的 1030 参考）；
教训固定：**评估器必须读 cfg 的 `use_bos` 决定是否移位，不能假设训练约定。**

#### 7.119.5 结论与下一步

1. **架构能拟合**（单图 CE 0.089 / 97.56%）⇒ 100 图的差距来自**容量-数据-优化**的组合，不是"不会学"；
2. **BOS 值得打开**（CE −0.64、一致率 +10pp），但它只修冷启动，不修 drift；
3. **自由采样的主因是 exposure bias + 过拟合**。对症方案按成本排序：
   - **免费**：推理侧用"教师强制引导"解码（给定真前缀 rollout、自一致性重排）先验证 drift 假说；
   - **便宜**：scheduled sampling / free-running 微调（用模型自己的输出做输入继续训几百步）；
   - **根本**：换更多数据（100 图 × 2 行 × 20 epoch 已在过拟合）。

#### 7.119.6 待办

- **exposure-bias 定量诊断**：同一批行上算 TF CE 2.131 与"自由 rollout 的自预测 CE"，两者之差
  就是 drift 的直接度量（比看图说话可靠）；
- scheduled sampling 的最小实现（先只在前 25% 位置用预测 token 替换真值，逐步扩大）；
- 单图实验的图已存 `i5build/of_single/`，100 图各 step 的 TF/FREE 图在 `i5build/ar_overfit100_bos/`。



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW_SECTION + '# 附录 A：AR 训练路径的实测证据', 'insert 7.119'),
    ('**本文是 118 节的【时序记录】', '**本文是 119 节的【时序记录】', 'header: 119'),
    ('**本报告的 118 节里有 12 节', '**本报告的 119 节里有 12 节', 'F header: 119'),
    ('| **GT 1030 等效目标（北极星）** | ★ **定义+判决矩阵+计分板；AR 步 1030 分数 0.21× → 目标 ≥0.84** | §7.118 · `gt1030/` |',
     '| **GT 1030 等效目标（北极星）** | ★ **定义+判决矩阵+计分板；AR 步 1030 分数 0.21× → 目标 ≥0.84** | §7.118 · `gt1030/` |\n'
     '| **单图过拟合 / BOS 判决** | ★ **单图 CE 0.089（97.6% 重构）；100 图 BOS CE 2.13 / 50.2%（无 BOS 2.77 / 40.3%）；自由采样仍糊 ⇒ exposure bias 主因** | §7.119 |',
     'H table: 7.119 row'),
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
