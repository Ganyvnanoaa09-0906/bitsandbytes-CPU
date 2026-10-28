# -*- coding: utf-8 -*-
"""dl_models.py — 从 hf-mirror 下载两个模型（本地现已可达）。

背景更正：旧的"HF 不可达"结论已作废 —— 重测显示 hf-mirror.com 与 modelscope.cn 都通
（1.2s / 0.7s），只有 huggingface.co 与 civitai.com 超时。

① SG161222/Realistic_Vision_V5.1_noVAE
   SD1.5 的**写实微调**（经典 photoreal 选择），单文件 safetensors。
   注意 "noVAE" ⇒ 复用 counterfeit-v3 的 VAE/CLIP（它本来就是 SD1.5 原版）,
   与 sd_refine_real.py 的拼装方式一致。
   ⇒ 用来替代 counterfeit-v3（动漫）做"写实 cos"方向的精修。

② Qwen/Qwen2-VL-2B-Instruct
   正好 2B 的**视觉语言模型**，一箭双雕：
   (a) 用户想拿它压测 bitsandbytes-CPU 那套 fork（此前只验证到 Qwen3.5-0.8B）；
   (b) 它能当 captioner，修我们那批"66 个 tag、自相矛盾"的烂标注。
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
os.environ.setdefault('HF_HOME', r'D:\work\hf_home')

from huggingface_hub import snapshot_download

JOBS = [
    ('realvis', 'SG161222/Realistic_Vision_V5.1_noVAE',
     ['*.json', '*.txt', '*.safetensors', '*.ckpt', '*.yaml', '*.md']),
    ('qwenvl2b', 'Qwen/Qwen2-VL-2B-Instruct',
     ['*.json', '*.txt', '*.safetensors', '*.py', '*.model', '*.jinja']),
]

which = sys.argv[1] if len(sys.argv) > 1 else 'all'
for tag, repo, pats in JOBS:
    if which not in ('all', tag):
        continue
    t0 = time.time()
    print('=== [%s] %s ===' % (tag, repo), flush=True)
    try:
        p = snapshot_download(repo, allow_patterns=pats,
                              local_dir=os.path.join(r'D:\work\textmodel', tag))
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(p) for f in fs)
        print('  OK -> %s  (%.2f GB, %.0f s, %.1f MB/s)'
              % (p, sz / 1e9, time.time() - t0, sz / 1e6 / max(1e-6, time.time() - t0)), flush=True)
    except Exception as e:
        print('  FAIL: %s' % str(e)[:200], flush=True)
