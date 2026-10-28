"""Download a matrix of tiny random HF models (different architectures) through hf-mirror.

The point is generality: the iGPU trainer must work for whatever architecture a user brings, not just
llama. These tiny models are a few MB each and exercise: llama/mistral/qwen2/gpt2/gptneox/falcon/bloom/
mixtral (MoE)/gemma/phi3/olmo/mpt.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import sys
import traceback

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

from huggingface_hub import snapshot_download

TARGET = r'D:\deepsleep\tiny_models'
MODELS = [
    'HuggingFaceH4/tiny-random-LlamaForCausalLM',
    'hf-internal-testing/tiny-random-MistralForCausalLM',
    'hf-internal-testing/tiny-random-Qwen2ForCausalLM',
    'sshleifer/tiny-gpt2',
    'hf-internal-testing/tiny-random-GPTNeoXForCausalLM',
    'hf-internal-testing/tiny-random-FalconForCausalLM',
    'hf-internal-testing/tiny-random-BloomForCausalLM',
    'hf-internal-testing/tiny-random-MixtralForCausalLM',
    'hf-internal-testing/tiny-random-GemmaForCausalLM',
    'hf-internal-testing/tiny-random-Phi3ForCausalLM',
    'hf-internal-testing/tiny-random-OlmoForCausalLM',
    'hf-internal-testing/tiny-random-MptForCausalLM',
]

os.makedirs(TARGET, exist_ok=True)
ok, fail = [], []
for mid in MODELS:
    local = os.path.join(TARGET, mid.replace('/', '__'))
    try:
        snapshot_download(repo_id=mid, local_dir=local, allow_patterns=[
            '*.json', '*.safetensors', '*.txt', '*.model', '*.bin'
        ])
        ok.append((mid, local))
        print('OK   ', mid, flush=True)
    except Exception as e:
        fail.append((mid, str(e)[:120]))
        print('FAIL ', mid, type(e).__name__, str(e)[:120], flush=True)

print()
print('downloaded %d, failed %d' % (len(ok), len(fail)))
for m, e in fail:
    print('  FAIL', m, e)
