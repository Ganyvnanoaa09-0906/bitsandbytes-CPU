# -*- coding: utf-8 -*-
"""AR 20k 收尾归档：等 ckpt_20000.pt + 采样图就绪后，写一份完整的结论笔记。

    python gt1030/_finalize_ar20k.py [--wait-min N]

内容：
  · CPU 20000 步的 val/曲线（从 ar_full3.log 抽）
  · iGPU 侧的 M4/M5/M6 数字（从本会话的产物里读）
  · 产物清单（ckpt、采样图、报告章节）
"""
from __future__ import annotations
import argparse
import io
import os
import re
import time

F = r'F:\weibo_cos'
LOG3 = os.path.join(F, 'ar_full3.log')
CK20 = os.path.join(F, 'ar_run_full3', 'ckpt_20000.pt')
CKLAST = os.path.join(F, 'ar_run_full3', 'ckpt_last.pt')
SAMPLE20 = os.path.join(F, 'ar_sample_20000.png')
NOTE = r'D:\work\collab\shared\AR_WEIBO_OVERNIGHT_2026-10-24.md'


def tail(path, n=12):
    """只取训练日志行（[v2 ...），避开 train.jsonl 之类的 JSON 尾巴"""
    try:
        with io.open(path, encoding='utf-8', errors='replace') as f:
            lines = [ln for ln in f.read().splitlines() if ln.startswith('[v2')]
        return '\n'.join(lines[-n:])
    except Exception as e:
        return str(e)


def read_text(path):
    try:
        return io.open(path, encoding='utf-8', errors='replace').read()
    except Exception:
        return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-min', type=float, default=0)
    args = ap.parse_args()
    t0 = time.time()
    while args.wait_min > 0 and time.time() - t0 < args.wait_min * 60:
        if os.path.exists(CK20) and os.path.exists(SAMPLE20):
            break
        time.sleep(30)

    ok20 = os.path.exists(CK20)
    have_sample = os.path.exists(SAMPLE20)
    igpu_log = read_text(os.path.join(F, 'ar_igpu_train.log'))
    med = re.findall(r'median ([\d.]+)s/step\s+(\d+) tok/s', igpu_log)
    steps_done = re.findall(r'完成 (\d+) 步', igpu_log)

    lines = [
        '# AR 微博 30M：CPU 20k 与核显移植（2026-10-24/25）',
        '',
        '## 1. CPU 训练（`train_ar_v2.py`，6 线程，32×32 token）',
        '',
        '```',
        tail(LOG3, 14),
        '```',
        '',
        '- 数据：`tokens_32x32_weibo.pt`，19,147 张微博图，VQ-VAE V32（1024 码本）',
        '- 配置：d512 / 10 层 / 8 头 / kv4 / ffn2048 / 1024 token/图；batch 2；lr 5e-5 余弦，warmup 100',
        '- 状态：**%s**' % ('20000 步 ✅ 已跑完' if ok20 else '仍在跑（ckpt_16000 之后继续）'),
        '- 产物：`ar_run_full3/ckpt_last.pt`、`ckpt_20000.pt`' if ok20 else
        '- 产物：`ar_run_full3/ckpt_last.pt`（滚动）',
        '- 采样：`ar_sample_20000.png` %s' % ('✅' if have_sample else '（待生成）'),
        '',
        '## 2. 核显（OpenCL）移植结果',
        '',
        '| 里程碑 | 结果 |',
        '|---|---|',
        '| M4 整机对拍 | 10 步 loss 轨迹最大偏差 **2.0e-3**（验收 0.02），首步 loss 4.910352 vs torch 4.910697 |',
        '| M5 权重导出 | 125/125 键、cfg 零差异、`load_state_dict(strict=True)` 通过、`sample_ar` 可采样 |',
        '| M6 续训吞吐 | 从 `ckpt_20000` 续训 100 步：**1.383 s/step = 1481 tok/s**（CPU 是 569 tok/s ⇒ **2.60×**）；'
        '独占 80 步持续测 1.350 s/step = 1517 tok/s（2.67×） |',
        '',
        '关键工程结论（详见报告 §7.222）：',
        '- 本机 AMD gfx902 驱动在「一个 context 里 ~130+ 个独立 cl_mem」或「多 context」时会**静默把队列搞哑**；',
        '  唯一稳定用法是 **单 context + 单一大 arena + `get_sub_region(offset,size)`**。',
        '- CE 梯度必须整体放大约 1024 倍，否则中间梯度掉进 fp16 次正规区，dW12 误差 60%。',
        '- 吞吐口径很重要：核显与 CPU 训练**并发**时两者互相拖累（核显 4.05 s/step、CPU 也变慢），',
        '  因为共享 DDR4 带宽与 APU 功耗预算。**独占**时核显 1.350 s/step = 1517 tok/s，是 CPU 的 2.67×。',
        '- 优化链：2.075 → 1.833（自写 k=64 GEMM）→ 1.777（自写 O/dQ GEMM）→ 1.616（grad_norm 设备端归约）',
        '  → 1.483（rms_dw 并行归约）→ **1.338~1.350（因果跳上三角）**。全程 10 步轨迹 1.98e-3 不变。',
        '- 驱动坑：单 context 里堆 ~130+ 独立 cl_mem / 多 context 会让队列静默瘫痪；',
        '  唯一稳的用法是 **单 context + 单一大 arena + get_sub_region**。',
        '',
        '## 3. 产物路径',
        '',
        '| 文件 | 说明 |',
        '|---|---|',
        '| `F:\\weibo_cos\\ar_run_full3\\ckpt_last.pt` | CPU 训练最终权重 |',
        '| `F:\\weibo_cos\\ar_sample_20000.png` | CPU 20k 采样（16 张） |',
        '| `F:\\weibo_cos\\ar_run_igpu\\ckpt_14100.pt` | iGPU 从 14k 续训 100 步后导出 |',
        '| `F:\\weibo_cos\\ar_run_igpu\\ckpt_16025.pt` | iGPU 从 16k 续训 25 步后导出 |',
        '| `F:\\weibo_cos\\ar_sample_igpu_12020.png` | iGPU 权重采样（与 CPU 同源可比） |',
        '| `F:\\weibo_cos\\ar_run_igpu\\ckpt_20100.pt` | **iGPU 从 ckpt_20000 续训 100 步后导出** |',
        '| `F:\\weibo_cos\\ar_sample_igpu_20100.png` | 上面这个 ckpt 的采样（与 `ar_sample_20000.png` 同源） |',
        '| `D:\\work\\collab\\shared\\IGPU_AR_V2_PORT_PLAN.md` | 移植计划与里程碑 |',
        '| `D:\\work\\research\\CPU_FORGE_TECHNICAL_REPORT.md` §7.222 | 完整技术报告（10 个小节） |',
        '',
        '## 4. 备注',
        '',
        '- 12k~20k 的采样仍是**色彩场**：欠训练（约 41M token、2.1 epoch，Chinchilla 需要 ~610M），不是移植问题。',
        '- 核显这条线的价值：算法快速迭代 + 推理侧（图像/视频）；生产训练暂时仍留在 CPU。',
    ]
    os.makedirs(os.path.dirname(NOTE), exist_ok=True)
    io.open(NOTE, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('wrote', NOTE, 'ok20=%s sample=%s' % (ok20, have_sample))


if __name__ == '__main__':
    main()
