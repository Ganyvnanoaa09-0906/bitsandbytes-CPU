# -*- coding: utf-8 -*-
"""追加 §7.222.8：iGPU 连续 100 步稳定性 + 吞吐（这是"能训练"的最直接证据）。"""
import io

TEXT = r"""

#### 7.222.8 连续 100 步稳定性 + 吞吐（能训练的最直接证据）

前面所有对拍都是"一步/十步"级别的。核显这条线真正的风险是**驱动长跑会不会中途哑掉**
（§7.222.2）。所以从 CPU trainer 的 `ckpt_14000.pt` 直接接着跑 100 步：

| 项 | 值 |
|---|---|
| 命令 | `python gt1030/ar_v2_train.py --ckpt ckpt_14000.pt --steps 100 --lr 2e-5` |
| 结果 | **100/100 步全部完成，无一次驱动故障、无一次上传校验失败** |
| 用时 | 449.2 s，median **4.361 s/step** |
| 吞吐 | **470 tok/s**（2048 token/步） |
| loss | 4.6338 → 6.1311（batch 间波动 3.58~6.13，与 CPU 侧 tail 的 3.8~5.5 同量级） |
| gnorm | 0.87~3.55（clip 1.0 生效） |
| 导出 | `F:\weibo_cos\ar_run_igpu\ckpt_14100.pt`（116MB，CPU 同格式，可直接采样/续训） |

**这解决了"能不能跑"**：单 context + 3GB arena 的方案在 100 步里没有任何一次静默坏掉。
**但"值不值得跑"仍然是否**：4.36s/step vs CPU 3.1s/step（而且这 100 步是与 CPU 训练同时跑、
互相抢资源的情况下测的；独占时前面测到 4.02s/step，仍然慢）。

**⇒ 结论定型**：核显这条线现在是**功能完备、数值对齐、可长跑**的训练实现，
唯一缺的是"快"——而那一步需要向量化 fp16 GEMM + flash attention（§7.222.6），属独立工程。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))
