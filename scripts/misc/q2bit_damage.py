"""2-bit mixed-precision quantization damage eval (Qwen3-1.7B)."""
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

import argparse
import gc
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
from scripts.lib import q2bit_mixed as Q2
from scripts.lib.q3bit_real_train import load_corpus, make_batches, loss_on, rss_gb

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--skip-names', default='')
    args = ap.parse_args()
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                     low_cpu_mem_usage=True, local_files_only=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                     low_cpu_mem_usage=True, local_files_only=True)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    vocab = int(getattr(model.config, 'vocab_size', 1024) or 1024)
    text = load_corpus()
    evals = make_batches(tok, text, args.seq, 8, seed=99, fallback_vocab=vocab)
    base = loss_on(model, evals)
    rss0 = rss_gb()
    skip = tuple(x.strip() for x in args.skip_names.split(',') if x.strip())
    t0 = time.perf_counter()
    n = Q2.convert_linears_2bit(model, block=args.block, skip_names=('lm_head',) + skip)
    conv_s = time.perf_counter() - t0
    gc.collect()
    after = loss_on(model, evals)
    print('Q2_EVAL base=%.4f after=%.4f damage=%.4f n=%d rss_base=%.2f rss=%.2f conv=%.1fs skip=%s'
          % (base, after, after - base, n, rss0, rss_gb(), conv_s, args.skip_names), flush=True)


if __name__ == '__main__':
    main()
