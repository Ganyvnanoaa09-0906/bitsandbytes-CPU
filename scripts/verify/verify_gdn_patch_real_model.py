# -*- coding: utf-8 -*-
"""verify_gdn_patch_real_model.py — end-to-end check of patch_transformers on Qwen3.5-0.8B.

Review question: the GDN CPU patch replaces transformers' torch fallback functions with the
fused recurrent kernel. Does the patched model actually take the patched path, and does it
produce the same output as the unpatched torch fallback?

Runs CPU-only, fp32, a short sequence. Reports max/mean abs diff of last_hidden_state and logits.
"""
from __future__ import annotations
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

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import torch  # noqa: E402

MODEL = r"D:\deepsleep\_test\models\qwen3.5-0.8B"


def main() -> int:
    torch.manual_seed(0)
    torch.set_num_threads(6)
    from transformers import Qwen3_5ForConditionalGeneration

    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("loaded in %.1fs | params %.3fB | dtypes %s" % (
        time.time() - t0, sum(p.numel() for p in model.parameters()) / 1e9,
        sorted({str(p.dtype) for p in model.parameters()})))

    ids = torch.randint(1, 1000, (2, 24))
    with torch.no_grad():
        out_unpatched = model.model(input_ids=ids).last_hidden_state
    print("unpatched forward OK", tuple(out_unpatched.shape))

    from bitsandbytes.gdn_cpu import patch_transformers
    patched = patch_transformers()
    print("patch_transformers ->", patched)
    import transformers.models.qwen3_next.modeling_qwen3_next as m_next
    for attr in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
        fn = getattr(m_next, attr, None)
        print("  %-32s patched=%s" % (attr, getattr(fn, "_gdn_cpu_patched", False)))
    if not patched:
        print("FAIL: patch_transformers returned False")
        return 1

    t0 = time.time()
    with torch.no_grad():
        out_patched = model.model(input_ids=ids).last_hidden_state
    print("patched forward OK in %.2fs" % (time.time() - t0))

    diff = (out_patched - out_unpatched).abs()
    denom = out_unpatched.abs().max().item()
    print("last_hidden_state: max|d| %.3e  mean|d| %.3e  (max|x| %.3f)" % (
        diff.max().item(), diff.mean().item(), denom))
    ok = diff.max().item() <= 1e-4 * max(denom, 1.0) + 2e-2
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
