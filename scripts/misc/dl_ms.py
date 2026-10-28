# -*- coding: utf-8 -*-
"""dl_ms.py — 测 ModelScope 通道的真实下载速度，并下拉一个 2B 级扩散模型。

为什么转向 ModelScope：
  hf-mirror 的大文件 302 到 cas-bridge.xethub.hf.co，实测 0.06~0.11 MB/s
  （SDXL UNet 5.14 GB 要十几小时）⇒ 不可用。
  而 modelscope.cn 的 API 0.4~0.7 s 响应 ⇒ 值得测真实吞吐。

用户目标："2b 的扩散模型拿去适配 bnb"（CPU 4-bit）。
  候选：SDXL（UNet 2.6B，未门禁，ModelScope 有镜像）、SD3-Medium（MMDiT 2B）。
  注意：2.6B fp16 = 5.2 GB，**4-bit ≈ 1.3 GB** ⇒ 16 GB 内存完全放得下；
  重点是【量化后能不能在 CPU 上跑起来】，不是速度。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import os, sys, time

os.environ.setdefault('MODELSCOPE_CACHE', r'D:\work\ms_cache')

MID = os.environ.get('MS_MODEL', 'AI-ModelScope/stable-diffusion-xl-base-1.0')

print('=== 目标: %s ===' % MID)
try:
    from modelscope.hub.api import HubApi
    api = HubApi()
    files = api.get_model_files(MID, recursive=True)
    tot = 0
    for f in files:
        s = f.get('Size') or 0
        tot += s
    print('  文件数 %d, 合计 %.2f GB' % (len(files), tot / 1e9))
    for f in sorted(files, key=lambda x: -(x.get('Size') or 0))[:6]:
        print('    %-58s %s' % (f.get('Path'), f.get('Size')))
except Exception as e:
    print('  列举失败: %s' % str(e)[:150])

print()
print('=== 真实下载测速（取一个中等文件，限时 60 s）===')
try:
    from modelscope.hub.file_download import model_file_download
    # 先挑一个 10~200 MB 的文件测速，避免直接下 5 GB
    cand = [f.get('Path') for f in files
            if (f.get('Size') or 0) > 5e6 and (f.get('Size') or 0) < 4e8]
    cand.sort(key=lambda p: len(p))
    print('  候选:', cand[:3])
    if cand:
        t0 = time.time()
        p = model_file_download(MID, cand[0], cache_dir=r'D:\work\ms_cache')
        dt = time.time() - t0
        sz = os.path.getsize(p)
        print('  OK -> %s' % p)
        print('  %.1f MB in %.1f s = %.2f MB/s  ⇒ 5 GB 约 %.1f 小时'
              % (sz / 1e6, dt, sz / 1e6 / max(1e-6, dt), 5000 / (sz / 1e6 / max(1e-6, dt)) / 3600))
except Exception as e:
    print('  下载失败: %s: %s' % (type(e).__name__, str(e)[:200]))
