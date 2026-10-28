"""q3bit_real_train.py — end-to-end 3-bit frozen base + LoRA on a real HF model.

Compares, in separate processes so memory numbers are honest:
  fp32 : frozen fp32 base + LoRA adapters on every nn.Linear
  q3bit: frozen 3-bit base (bitsandbytes.q3bit) + the same LoRA adapters

Reports: quantization damage (eval loss before/after conversion), training loss,
step time, RSS (baseline / after conversion / peak), trainable params.

Usage:
  python q3bit_real_train.py --model D:\\deepsleep\\qwen3-1.7B --mode q3bit --steps 100
  python q3bit_real_train.py --model D:\\deepsleep\\qwen3-1.7B --mode fp32  --steps 100
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time

import numpy
import torch
import torch.nn
import torch.nn.functional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bitsandbytes'))
sys.path.insert(0, HERE)

import bitsandbytes
from bitsandbytes import q3bit  # noqa: E402

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None


def rss_gb():
    if psutil is None:
        return float('nan')
    return psutil.Process().memory_info().rss / 1e9


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- LoRA (fp32 baseline)
class LoRALinearFP32(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # reuse the original tensors (no clone: cloning doubles the base model RSS)
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale
        return base + lora


def wrap_lora_fp32(model: nn.Module, r: int, alpha: int, skip=('lm_head',)) -> int:
    n = 0

    def walk(mod, prefix=''):
        nonlocal n
        for name, child in list(mod.named_children()):
            full = f'{prefix}.{name}' if prefix else name
            if isinstance(child, nn.Linear):
                if name in skip:
                    continue
                setattr(mod, name, LoRALinearFP32(child, r, alpha))
                n += 1
            elif isinstance(child, (LoRALinearFP32, q3bit.QuantLinear3bit)):
                continue
            else:
                walk(child, full)
    walk(model)
    return n


# ---------------------------------------------------------------- data
def load_corpus() -> str:
    paths = [
        os.path.join(HERE, 'bitsandbytes', 'README.md'),
        os.path.join(HERE, 'bitsandbytes', 'docs_cpu', 'QUICKSTART.md'),
        os.path.join(HERE, 'bitsandbytes', 'docs_cpu', 'TECHNICAL_GUIDE.md'),
        os.path.join(HERE, 'README.md'),
        r'D:\work\research\CPU_FORGE_TECHNICAL_PAPER.md',
    ]
    chunks = []
    for p in paths:
        try:
            chunks.append(open(p, encoding='utf-8', errors='replace').read())
        except OSError:
            pass
    return '\n\n'.join(chunks)[:3_000_000]


def make_batches(tok, text: str, seq: int, count: int, seed: int, fallback_vocab: int = 1024):
    ids = tok(text, return_tensors='np', add_special_tokens=False)['input_ids'].reshape(-1)
    if len(ids) < seq + 2:  # some tiny test tokenizers are empty dummies
        print('  WARNING: tokenizer produced %d tokens; falling back to random token ids' % len(ids),
              flush=True)
        rng0 = np.random.default_rng(seed + 7)
        ids = rng0.integers(0, max(2, fallback_vocab), size=seq * 64 + 1024).astype(np.int64)
        rng = np.random.default_rng(seed)
        max_start = max(1, len(ids) - seq - 2)
        return [torch.from_numpy(ids[int(rng.integers(0, max_start)):][:seq + 1]).unsqueeze(0)
                for _ in range(count)]
    rng = np.random.default_rng(seed)
    max_start = len(ids) - seq - 2
    out = []
    for _ in range(count):
        s = int(rng.integers(0, max_start))
        chunk = ids[s:s + seq + 1].astype(np.int64)
        out.append(torch.from_numpy(chunk).unsqueeze(0))
    return out


def loss_on(model, batches, limit=4):
    model.eval()
    vals = []
    with torch.no_grad():
        for b in batches[:limit]:
            # HF CausalLM shifts internally: pass the same tokens as input and labels
            out = model(input_ids=b[:, :-1], labels=b[:, :-1], use_cache=False)
            vals.append(float(out.loss))
    model.train()
    return float(np.mean(vals))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--mode', choices=['fp32', 'q3bit'], required=True)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lora-alpha', type=int, default=16)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--out', default='')
    ap.add_argument('--skip-names', default='')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--optim', default='adamw8bit', choices=['adamw8bit','adamw'])
    ap.add_argument('--save-every', type=int, default=0)
    ap.add_argument('--resume', default='')
    args = ap.parse_args()

    torch.set_num_threads(6)
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True)
    model.config.use_cache = False
    total = sum(p.numel() for p in model.parameters())
    base_rss = rss_gb()
    log('model %s | %.2f B params | load %.1f s | RSS %.2f GB'
        % (args.model, total / 1e9, time.perf_counter() - t0, base_rss))

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    text = load_corpus()
    vocab = int(getattr(model.config, 'vocab_size', 1024) or 1024)
    train = make_batches(tok, text, args.seq, args.steps * args.batch + 8, seed=1,
                         fallback_vocab=vocab)
    evals = make_batches(tok, text, args.seq, 8, seed=99, fallback_vocab=vocab)
    log('corpus: %.2f MB chars | train batches %d | eval %d | seq %d'
        % (len(text) / 1e6, len(train), len(evals), args.seq))

    base_loss = loss_on(model, evals)
    log('eval loss before conversion: %.4f' % base_loss)

    t1 = time.perf_counter()
    if args.mode == 'q3bit':
        _skip = tuple(x.strip() for x in args.skip_names.split(',') if x.strip())
        n = q3bit.convert_linears_3bit(model, block=args.block, lora_r=args.lora_r,
                                       lora_alpha=args.lora_alpha,
                                       skip_names=('lm_head',) + _skip)
    else:
        n = wrap_lora_fp32(model, args.lora_r, args.lora_alpha)
    conv_time = time.perf_counter() - t1
    gc.collect()
    conv_rss = rss_gb()
    quant_loss = loss_on(model, evals) if args.mode == 'q3bit' else base_loss
    log('converted %d linears in %.1f s | RSS %.2f GB | eval after conversion: %.4f'
        % (n, conv_time, conv_rss, quant_loss))

    if args.eval_only:
        print('EVAL_ONLY base=%.4f after=%.4f damage=%.4f n=%d skip=%s rss=%.2f' % (base_loss, quant_loss, quant_loss-base_loss, n, args.skip_names, conv_rss), flush=True)
        return 0
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    log('trainable: %.2f M (%.2f%% of base) | %d tensors'
        % (trainable / 1e6, 100 * trainable / total, len(params)))
    if args.optim == 'adamw':
        opt = torch.optim.AdamW(params, lr=args.lr)
        opt_name = 'AdamW'
    else:
        try:
            opt = bnb.optim.AdamW8bit(params, lr=args.lr)
            opt_name = 'AdamW8bit'
        except Exception:
            opt = torch.optim.AdamW(params, lr=args.lr)
            opt_name = 'AdamW'
    log('optimizer: %s' % opt_name)

    start_step = 0
    ckpt_path = (args.out + '.ckpt.pt') if args.out else ''
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
        model.load_state_dict(ck['lora'], strict=False)
        try:
            opt.load_state_dict(ck['opt'])
        except Exception as e:
            log('optimizer state not restored: %s' % e)
        start_step = int(ck.get('step', 0))
        log('resumed %s @ step %d' % (args.resume, start_step))

    def save_ckpt(step):
        if not ckpt_path:
            return
        lora = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
        torch.save({'lora': lora, 'opt': opt.state_dict(), 'step': step}, ckpt_path)
        log('checkpoint %s @ step %d' % (ckpt_path, step))

    model.train()
    losses, times, rss_peak = [], [], conv_rss
    t_start = time.perf_counter()
    for step in range(start_step, args.steps):
        batch = train[step] if step < len(train) else train[-1]
        t0 = time.perf_counter()
        out = model(input_ids=batch[:, :-1], labels=batch[:, :-1], use_cache=False)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        losses.append(float(loss.detach()))
        times.append(dt)
        rss_peak = max(rss_peak, rss_gb())
        if step % args.log_every == 0 or step == args.steps - 1:
            log('  step %4d | loss %.4f | %.3f s | RSS %.2f GB'
                % (step, losses[-1], dt, rss_gb()))
        if not math.isfinite(losses[-1]):
            log('  !! non-finite loss, stopping')
            break
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(step + 1)
    save_ckpt(args.steps)
    total_s = time.perf_counter() - t_start
    med = statistics.median(times[1:]) if len(times) > 1 else times[0]
    final_loss = loss_on(model, evals)
    result = {
        'model': args.model, 'mode': args.mode, 'params_b': total / 1e9,
        'linears_converted': n, 'trainable_m': trainable / 1e6,
        'steps': len(losses), 'seq': args.seq, 'batch': args.batch,
        'lr': args.lr, 'lora_r': args.lora_r, 'block': args.block,
        'optimizer': opt_name,
        'rss_base_gb': base_rss, 'rss_after_gb': conv_rss, 'rss_peak_gb': rss_peak,
        'eval_loss_base': base_loss, 'eval_loss_after_convert': quant_loss,
        'eval_loss_final': final_loss,
        'train_loss_first': losses[0], 'train_loss_last': losses[-1],
        'train_loss_min': min(losses),
        'step_s_median': med, 'total_s': total_s,
        'load_and_convert_s': time.perf_counter() - t0 if False else None,
    }
    log('RESULT ' + json.dumps({k: v for k, v in result.items() if v is not None}, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, indent=1, ensure_ascii=False)
        log('wrote %s' % args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
