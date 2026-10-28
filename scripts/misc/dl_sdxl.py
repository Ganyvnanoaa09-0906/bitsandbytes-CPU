# -*- coding: utf-8 -*-
"""dl_sdxl.py — 从 ModelScope 下 SDXL base 1.0（只取 fp16 + 配置，避开 fp32 重复）。

为什么是 SDXL：
  · UNet 2.6B —— 正是用户说的"2b 级扩散模型拿去适配 bnb"的合适对象
    （此前 bnb 压测只到 Qwen3.5-0.8B；虽说 8B 也压过，但那是 LLM，
     扩散模型的算子构成不同：conv + attention + GroupNorm，量化行为需重测）；
  · 同时它是**写实能力**远超 SD1.5 的开放模型 —— 直接对应"写实 cos"那个质量档需求。

内存账（16 GB 机器）：
  fp16 全量 ~7 GB；**4-bit 后 UNet 约 1.3 GB + 两个 text encoder 约 0.4 GB** ⇒ 宽裕。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import os, sys, time

os.environ.setdefault('MODELSCOPE_CACHE', r'D:\work\ms_cache')
MID = 'AI-ModelScope/stable-diffusion-xl-base-1.0'

from modelscope.hub.api import HubApi
api = HubApi()
files = api.get_model_files(MID, recursive=True)
print('仓库文件 %d 个:' % len(files))
for f in files:
    sz = f.get('Size') or 0
    print('  %-62s %10.1f MB' % (f.get('Path'), sz / 1e6))

# 只要 fp16 + 配置，避开 fp32 重复（省一半空间）
want = [f.get('Path') for f in files
        if (f.get('Size') or 0) > 0
        and ('fp16' in f.get('Path')
             or f.get('Path').endswith(('.json', '.txt', '.model')))]
tot = sum((f.get('Size') or 0) for f in files if f.get('Path') in want)
print('\n选定 %d 个文件, 合计 %.2f GB' % (len(want), tot / 1e9))

from modelscope.hub.file_download import model_file_download
t0 = time.time()
done = 0
for p in want:
    try:
        fp = model_file_download(MID, p, cache_dir=r'D:\work\ms_cache')
        done += os.path.getsize(fp)
        if done % (200 * 1e6) < 50 * 1e6 or 'unet' in p:
            print('  %-58s 累计 %.2f GB  %.1f s' % (p, done / 1e9, time.time() - t0), flush=True)
    except Exception as e:
        print('  FAIL %s: %s' % (p, str(e)[:120]))
print('\n完成 %.2f GB, 用时 %.1f 分钟' % (done / 1e9, (time.time() - t0) / 60))
