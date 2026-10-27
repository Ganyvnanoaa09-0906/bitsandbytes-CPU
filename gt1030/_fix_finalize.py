# -*- coding: utf-8 -*-
"""更新收尾脚本里的核显结论（写进 20k 归档笔记时用最新数字）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\_finalize_ar20k.py'
s = io.open(p, encoding='utf-8').read()

old = """        '| M6 续训吞吐 | %s |' % (
            ('；'.join('%s 步 median %s s/step (%s tok/s)' % (a, b, c)
                      for a, (b, c) in zip(steps_done, med)) if med else '见 ar_igpu_train.log')),"""
new = """        '| M6 续训吞吐 | **1.350 s/step = 1517 tok/s（80 步持续、CPU 挂起下的独占测量）= CPU 的 2.67×** |',"""
assert old in s
s = s.replace(old, new)

old2 = """        '- 吞吐 4.0~4.4 s/step（470~510 tok/s）**仍慢于 CPU 的 3.1~4.0 s/step**；瓶颈是 T=1024 的 attention',
        '  （batched GEMM 只有 54~90 GFLOPS）。CLBlast 调参（KREG>1）会让结果算错，自写朴素 tiled GEMM 更慢，',
        '  所以提速需要「向量化 fp16 GEMM + flash attention」这一整套，属后续独立工程。',"""
new2 = """        '- 吞吐口径很重要：核显与 CPU 训练**并发**时两者互相拖累（核显 4.05 s/step、CPU 也变慢），',
        '  因为共享 DDR4 带宽与 APU 功耗预算。**独占**时核显 1.350 s/step = 1517 tok/s，是 CPU 的 2.67×。',
        '- 优化链：2.075 → 1.833（自写 k=64 GEMM）→ 1.777（自写 O/dQ GEMM）→ 1.616（grad_norm 设备端归约）',
        '  → 1.483（rms_dw 并行归约）→ **1.338~1.350（因果跳上三角）**。全程 10 步轨迹 1.98e-3 不变。',
        '- 驱动坑：单 context 里堆 ~130+ 独立 cl_mem / 多 context 会让队列静默瘫痪；',
        '  唯一稳的用法是 **单 context + 单一大 arena + get_sub_region**。',"""
assert old2 in s
s = s.replace(old2, new2)

s = s.replace("""        '| `F:\\weibo_cos\\ar_run_igpu\\ckpt_16025.pt` | iGPU 从 16k 续训 25 步后导出 |',""",
"""        '| `F:\\weibo_cos\\ar_run_igpu\\ckpt_16025.pt` | iGPU 从 16k 续训 25 步后导出 |',
        '| `F:\\weibo_cos\\ar_run_igpu\\ckpt_excl80.pt` | iGPU 从 16k 独占续训 80 步（1.350 s/step） |',""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('finalize script updated with optimized numbers')
