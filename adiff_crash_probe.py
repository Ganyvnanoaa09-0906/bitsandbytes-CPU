# -*- coding: utf-8 -*-
"""adiff_crash_probe.py — 定位 AnimateDiff 段错误的确定性触发条件

背景（实测，全部为 0xC0000005）:
    10:12  c10.dll         fault offset 0x90514   ← 去掉 .contiguous() 的 A/B
    11:01  torch_cpu.dll   fault offset 0x8e4a269
    11:10  VCRUNTIME140.dll
    11:14  c10.dll         fault offset 0x90514
    11:23  c10.dll         fault offset 0x90514   ← 又一个

`0x90514` 在 c10.dll 里**反复出现** ⇒ 怀疑**确定性**，而不是单纯内存竞争。
（我先前归因于"多进程抢内存"，但 free 有 10 GB，且同一 offset 复现，这个解释站不住。）

本脚本逐一分离变量，找出到底哪一步崩：
    1) 只 build()                     —— 加载本身是否崩
    2) build() + 单次 UNet 前向        —— 前向是否崩
    3) build() + 2 步小规模完整采样     —— 采样是否崩
    4) 第 3 步 + 关闭 attention slicing
每一步都独立进程执行（子进程），崩了不影响下一步，且能精确定位。

用法: python adiff_crash_probe.py            # 逐阶段跑
      python adiff_crash_probe.py --stage 2  # 只跑某个阶段
"""
from __future__ import annotations

import os
import subprocess
import sys

STAGES = {
    1: ('build only', r'''
import sys, torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from anime_adiff import build
torch.set_num_threads(6)
p = build(scheduler='dpm')
print('STAGE1 OK: unet params = %.0fM' % (sum(x.numel() for x in p.unet.parameters())/1e6))
'''),
    2: ('build + one UNet forward', r'''
import sys, torch, time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from anime_adiff import build
torch.set_num_threads(6)
p = build(scheduler='dpm'); p.set_progress_bar_config(disable=True)
F, R = 6, 192
lat = torch.randn(1, 4, F, R//8, R//8)
ctx = torch.randn(1, 77, 768).repeat(F, 1, 1)
t = torch.tensor([500])
with torch.inference_mode():
    p.unet(lat, t, encoder_hidden_states=ctx)
    t0 = time.perf_counter(); y = p.unet(lat, t, encoder_hidden_states=ctx); dt = time.perf_counter()-t0
print('STAGE2 OK: unet fwd %.2f s, out %s' % (dt, tuple(y.sample.shape)))
'''),
    3: ('build + full 2-step sample (slicing DEFAULT)', r'''
import sys, torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from anime_adiff import build, ANIME_PROMPT
from diffusers import DPMSolverMultistepScheduler
torch.set_num_threads(6)
p = build(scheduler='dpm'); p.set_progress_bar_config(disable=True)
p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
g = torch.Generator(device='cpu').manual_seed(0)
with torch.inference_mode():
    o = p(ANIME_PROMPT, num_frames=6, num_inference_steps=2, guidance_scale=7.5,
          width=192, height=192, generator=g)
fr = o.frames[0]
print('STAGE3 OK: %d frames, first size %s' % (len(fr), fr[0].size))
'''),
    4: ('build + full 2-step sample (slicing OFF)', r'''
import sys, torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from anime_adiff import build, ANIME_PROMPT
from diffusers import DPMSolverMultistepScheduler
torch.set_num_threads(6)
p = build(scheduler='dpm'); p.set_progress_bar_config(disable=True)
try: p.disable_attention_slicing()
except Exception: pass
p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
g = torch.Generator(device='cpu').manual_seed(0)
with torch.inference_mode():
    o = p(ANIME_PROMPT, num_frames=6, num_inference_steps=2, guidance_scale=7.5,
          width=192, height=192, generator=g)
fr = o.frames[0]
print('STAGE4 OK: %d frames (slicing OFF), first size %s' % (len(fr), fr[0].size))
'''),
}


def main():
    only = None
    if '--stage' in sys.argv:
        only = int(sys.argv[sys.argv.index('--stage') + 1])
    print('=' * 80)
    print('AnimateDiff 段错误：逐阶段定性（每阶段独立子进程）')
    print('=' * 80)
    for n in sorted(STAGES):
        if only and n != only:
            continue
        label, code = STAGES[n]
        print('\n--- stage %d: %s ---' % (n, label), flush=True)
        p = subprocess.run([sys.executable, '-c', code],
                           capture_output=True, text=True, timeout=1800)
        rc = p.returncode
        tail = [l for l in (p.stdout or '').splitlines() if l.strip()][-3:]
        for l in tail:
            print('    %s' % l[:110])
        if rc == 0:
            print('    >>> rc=0  PASS')
        elif rc == -1073741819 or rc == 3221225477:
            print('    >>> rc=%d  0xC0000005 ACCESS VIOLATION' % rc)
        else:
            err = [l for l in (p.stderr or '').splitlines() if l.strip()][-2:]
            print('    >>> rc=%d  %s' % (rc, ' | '.join(x[:80] for x in err)))
    print('\n' + '=' * 80)
    print('读法: 第一个 FAIL 的阶段就是触发点；后续阶段不必再看。')
    print('=' * 80)


if __name__ == '__main__':
    main()
