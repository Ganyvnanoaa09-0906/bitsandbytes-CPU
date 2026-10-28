# -*- coding: utf-8 -*-
"""dl_probe.py — 实测两条下载通道能不能真拿到字节（不是只看元数据）。

已知：
  · hf-mirror.com 的 resolve URL 返回 **302 → cas-bridge.xethub.hf.co**（HF 的 Xet CDN）
  · 元数据 API 通，但 huggingface_hub 的 snapshot_download 报 ConnectTimeout
  ⇒ 要分清是"CDN 不可达"还是"hf_xet 传输层的问题"。
    对策：禁用 hf_xet（HF_HUB_DISABLE_XET=1），强制走经典 HTTP 直连。

  · modelscope SDK 1.38.1 已装，其 API 返回 200 ⇒ 备用通道。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys, time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'          # 关键：绕开 xet 传输层
os.environ.setdefault('HF_HOME', r'D:\work\hf_home')

import httpx

REPO = 'stabilityai/stable-diffusion-xl-base-1.0'
SMALL = 'model_index.json'
BIG = 'unet/diffusion_pytorch_model.fp16.safetensors'


def probe(url, nbytes=1 << 20, label=''):
    """真读 nbytes 字节，测实际吞吐。"""
    t0 = time.time()
    try:
        with httpx.stream('GET', url, follow_redirects=True, timeout=30.0) as r:
            print('  %-28s status=%s  final_host=%s' % (label, r.status_code, r.url.host))
            got = 0
            for chunk in r.iter_bytes(1 << 16):
                got += len(chunk)
                if got >= nbytes:
                    break
            dt = time.time() - t0
            print('    读 %d 字节  用时 %.1fs  ⇒ %.2f MB/s  (Content-Length=%s)'
                  % (got, dt, got / 1e6 / max(1e-6, dt), r.headers.get('content-length')))
            return True
    except Exception as e:
        print('  %-28s FAIL %s: %s' % (label, type(e).__name__, str(e)[:90]))
        return False


print('=== ① hf-mirror 小文件 ===')
probe('https://hf-mirror.com/%s/resolve/main/%s' % (REPO, SMALL), 1 << 19, 'model_index.json')
print('=== ② hf-mirror 大文件（真读 1 MB）===')
probe('https://hf-mirror.com/%s/resolve/main/%s' % (REPO, BIG), 1 << 20, 'sdxl unet fp16')
print('=== ③ 直连 xethub CDN（跳过镜像）===')
try:
    import huggingface_hub  # noqa
    print('   (需要 302 的签名 URL，见上一步输出)')
except Exception:
    pass
print('=== ④ modelscope 通道 ===')
try:
    from modelscope.hub.api import HubApi
    api = HubApi()
    mid = 'AI-ModelScope/stable-diffusion-xl-base-1.0'
    files = api.get_model_files(mid, recursive=False)
    print('  MS 文件数 %d:' % len(files))
    for f in files[:8]:
        print('    ', f.get('Path'), f.get('Size'))
except Exception as e:
    print('  MS FAIL %s: %s' % (type(e).__name__, str(e)[:120]))
