"""Does 4-bit work on an LLM, and how far can it be pushed? Measure it properly.

The diffusion work concluded that 2-bit destroys the output and that the fix has to be
COVERAGE (quantize the convolutions nothing was quantizing) rather than fewer bits. That
conclusion came from SD1.5, which is structurally the worst possible testbed for it: 68.5%
of its parameters are convolutions that bitsandbytes cannot reach, and its error compounds
through 20-50 sequential sampling steps.

An LLM is the opposite case on all three counts, which makes it the place where extreme
quantization has its best chance:

  * decode is memory-bandwidth bound, so fewer bytes per weight IS fewer seconds per
    token -- the thing diffusion could never buy (measured 1.01-1.05x there)
  * a slightly wrong logit still produces a fine token; diffusion's error compounds
  * the ecosystem ships Q4_K_M by default and reserves 2-bit for models that would
    otherwise not fit at all

The metric is perplexity on WikiText-2, the standard corpus, because a perplexity is an
average over many tokens and does not degenerate the way a per-element relative error
does. Decode speed is reported separately: they are different currencies and this project
has confused them before.

Usage:
    python llm_quant_ab.py --model D:\\work\\llm_env\\Qwen3-1.7B
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

torch.set_grad_enabled(False)


# --- quantized Linear --------------------------------------------------------------

def _pack4(codes):
    n = codes.numel()
    if n % 2:
        codes = torch.cat([codes, codes.new_zeros(1)])
    u = (codes.to(torch.int16) + 8).to(torch.uint8)
    return (u[0::2] << 4) | u[1::2], n


def _unpack4(packed, n):
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    out = torch.empty(hi.numel() * 2, dtype=torch.int16)
    out[0::2] = hi.to(torch.int16)
    out[1::2] = lo.to(torch.int16)
    return out[:n] - 8


def _fit_codebook(w, block, levels, iters=20, sample=200_000):
    """Lloyd-Max over per-block-normalized weights: the non-uniform 4-level codebook."""
    flat = w.reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.reshape(-1, block)
    scale = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    x = (blk / scale).reshape(-1)
    if x.numel() > sample:
        x = x[:: x.numel() // sample]
    c = torch.quantile(x, torch.linspace(0, 1, levels + 2)[1:-1]).sort().values
    for _ in range(iters):
        idx = (x.unsqueeze(1) - c.unsqueeze(0)).abs().argmin(dim=1)
        for k in range(levels):
            m = idx == k
            if m.any():
                c[k] = x[m].mean()
        c = c.sort().values
    return c


class QLinear(nn.Module):
    """Linear whose weight is held as integer codes plus one fp32 scale per block."""

    def __init__(self, lin, bits=4, block=64, codebook=False):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bits = bits
        self.block = block
        self.has_bias = lin.bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        if codebook:
            self.codebook = _fit_codebook(lin.weight.detach(), block, 1 << bits)
            self.qmax = None
        else:
            self.codebook = None
            self.qmax = (1 << (bits - 1)) - 1
        w = lin.weight.detach().reshape(-1)
        # values are linear in the first dimension, so a per-ROW scale is also needed for
        # a Linear: quantize in row-major blocks exactly like the conv path does
        self._quantize(w)

    def _quantize(self, flat):
        n = flat.numel()
        pad = (-n) % self.block
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blk = flat.reshape(-1, self.block)
        scale = (blk.abs().amax(dim=1).clamp_min(1e-12) / (self.qmax or 1)).to(torch.float32)
        if self.codebook is not None:
            norm = (blk / scale.unsqueeze(1)).reshape(-1)
            idx = (norm.unsqueeze(1) - self.codebook.unsqueeze(0)).abs().argmin(dim=1)
            code = idx.reshape(-1)[:n]
        else:
            code = torch.clamp(torch.round(blk / scale.unsqueeze(1)),
                               -self.qmax, self.qmax).reshape(-1)[:n]
        self.n_codes = n
        self.register_buffer("scales", scale)
        if self.codebook is not None:
            # codebook indices are unsigned and fit in `bits`
            self.register_buffer("codes", code.to(torch.uint8))
        elif self.bits == 8:
            self.register_buffer("codes", (code.to(torch.int16) + 128).to(torch.uint8))
        else:
            packed, _ = _pack4(code.to(torch.int8))
            self.register_buffer("codes", packed)

    def weight_dequant(self):
        n = self.n_codes
        if self.codebook is not None:
            c = self.codebook[self.codes.long()]
        elif self.bits == 8:
            c = (self.codes.to(torch.int16) - 128).to(torch.float32)
        else:
            c = _unpack4(self.codes, n).to(torch.float32)
        nb = self.scales.numel()
        need = nb * self.block
        if c.numel() < need:
            c = torch.cat([c, c.new_zeros(need - c.numel())])
        w = (c[:need].reshape(nb, self.block) * self.scales.unsqueeze(1)).reshape(-1)[:n]
        return w.reshape(self.out_features, self.in_features)

    def forward(self, x):
        w = self.weight_dequant()
        # The model runs in bf16, so a dequantised fp32 weight has to be cast back before
        # the matmul: F.linear refuses mixed dtypes outright rather than promoting.
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        return F.linear(x, w, self.bias)


def replace_linears(model, factory):
    """Replace every nn.Linear under `model`. Returns the count."""
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, factory(child))
            n += 1
        else:
            n += replace_linears(child, factory)
    return n


def quant_bytes(model):
    tot = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            tot += m.codes.numel() * m.codes.element_size()
            tot += m.scales.numel() * m.scales.element_size()
        if isinstance(m, bnb.nn.Linear4bit) and getattr(m.weight, "quant_state", None):
            tot += m.weight.numel() * m.weight.element_size()
            st = m.weight.quant_state
            for k in ("absmax", "quant_map", "nested_absmax", "nested_quant_map"):
                v = getattr(st, k, None)
                if torch.is_tensor(v):
                    tot += v.numel() * v.element_size()
    return tot


# --- measurement ------------------------------------------------------------------

def make_windows(tokenizer, path, n_windows, length):
    text = open(path, encoding="utf-8").read()
    ids = tokenizer(text)["input_ids"]
    need = n_windows * length
    if len(ids) < need:
        raise SystemExit(f"corpus has {len(ids)} tokens, need {need}")
    return [ids[i * length:(i + 1) * length] for i in range(n_windows)]


def perplexity(model, windows):
    total_nll, total_tok = 0.0, 0
    for w in windows:
        ids = torch.tensor([w], dtype=torch.long)
        out = model(ids, labels=ids)
        total_nll += out.loss.item() * (len(w) - 1)
        total_tok += len(w) - 1
    return math.exp(total_nll / total_tok)


def decode_speed(model, tokenizer, new_tokens=32):
    ids = tokenizer("The history of the Roman Empire", return_tensors="pt").input_ids
    t0 = time.perf_counter()
    model.generate(ids, max_new_tokens=new_tokens, do_sample=False,
                   use_cache=True, pad_token_id=tokenizer.eos_token_id or 0)
    dt = time.perf_counter() - t0
    return new_tokens / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\work\llm_env\Qwen3-1.7B")
    ap.add_argument("--corpus", default=r"D:\work\llm_env\data\wikitext2_test.txt")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--decode-tokens", type=int, default=32)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    global bnb
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"model  : {a.model}")
    print(f"corpus : {a.corpus}  ({a.windows} windows x {a.length} tokens)")
    print(flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    windows = make_windows(tok, a.corpus, a.windows, a.length)
    print(f"  {sum(len(w) for w in windows)} tokens prepared", flush=True)

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        m.eval()
        m.config.use_cache = True
        return m

    configs = [
        ("bf16 (reference)", None),
        ("bnb nf4 4bit (shipped)", "bnb4"),
        ("uniform 8bit/64", dict(bits=8, block=64)),
        ("uniform 4bit/64", dict(bits=4, block=64)),
        ("uniform 3bit/64", dict(bits=3, block=64)),
        ("uniform 2bit/64", dict(bits=2, block=64)),
        ("codebook 2bit/64", dict(bits=2, block=64, codebook=True)),
    ]

    n_params = None
    rows = []
    for label, cfg in configs:
        t0 = time.time()
        model = fresh()
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  model {n_params/1e9:.3f} B params, "
                  f"bf16 = {n_params*2/2**30:.2f} GiB", flush=True)
        nq = 0
        if cfg == "bnb4":
            def factory(lin):
                q = bnb.nn.Linear4bit(lin.in_features, lin.out_features,
                                      bias=lin.bias is not None,
                                      compute_dtype=torch.float32, quant_type="nf4",
                                      compress_statistics=True)
                q.weight = bnb.nn.Params4bit(lin.weight.data, requires_grad=False,
                                             quant_type="nf4",
                                             compress_statistics=True)
                # .to() is what actually performs the quantization: assigning a
                # Params4bit leaves the module with no quant_state, and the first
                # forward then fails inside fix_4bit_weight_quant_state_from_module.
                q.weight = q.weight.to("cpu")
                if lin.bias is not None:
                    q.bias = nn.Parameter(lin.bias.detach().clone())
                return q
            nq = replace_linears(model, factory)
        elif cfg:
            nq = replace_linears(model, lambda lin: QLinear(lin, **cfg))
        gc.collect()

        ppl = perplexity(model, windows)
        tps = decode_speed(model, tok, a.decode_tokens)
        qb = quant_bytes(model)
        rows.append((label, ppl, tps, qb, nq))
        print(f"  {label:<24} ppl {ppl:8.4f}   decode {tps:6.2f} tok/s   "
              f"quantized {qb/2**20:7.1f} MiB   {nq} layers   "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        gc.collect()

    base = rows[0][1]
    print(f"\n{'config':<24} {'ppl':>9} {'vs bf16':>9} {'tok/s':>8} {'speed':>8} "
          f"{'MiB':>9}")
    for label, ppl, tps, qb, nq in rows:
        print(f"{label:<24} {ppl:9.4f} {ppl/base:8.2f}x {tps:8.2f} "
              f"{tps/rows[0][2]:7.2f}x {qb/2**20:9.1f}")

    with open("llm_quant_ab_results.json", "w") as fh:
        json.dump([{"config": r[0], "ppl": r[1], "tok_s": r[2], "bytes": r[3],
                    "layers": r[4]} for r in rows], fh, indent=1)
    print("\nwrote llm_quant_ab_results.json")


if __name__ == "__main__":
    main()
