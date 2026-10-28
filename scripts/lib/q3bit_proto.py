# -*- coding: utf-8 -*-
"""q3bit_proto.py — 2/3-bit blockwise quantization prototype for CPU training.

Motivation: on a 15.4 GB CPU-only laptop, a fp32 0.8B model costs 3.2 GB of weights before
gradients/optimizer. Blockwise 3-bit weights (~0.35 GB) + LoRA can fit much larger models.
This prototype answers two questions with measurements instead of guesses:
  1. what is the quantization error of 2/3-bit versus 8-bit/fp32 (on real model weights)?
  2. can we actually *train* (LoRA) through a 2/3-bit frozen base on CPU, and what does it cost
     in speed and memory?

Modes:
  packtest : quantize/dequantize round-trip + error vs 8-bit / fp32 on random and real weights
  train    : replace all nn.Linear in the text model with QuantLinear (2/3/32-bit) + LoRA,
             run steps, report loss / step time / RSS / packed size

Usage:
  python q3bit_proto.py --mode packtest
  python q3bit_proto.py --mode train --model tiny --bits 3 --steps 20
  python q3bit_proto.py --mode train --model qwen --bits 3 --steps 1
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
import os
import sys
import time

import torch
import torch.nn
import torch.nn.functional

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

_KERNEL = None
_KERNEL_TRIED = False


def load_kernel():
    """Load the optional C dequant kernel (q3bit_kernel.dll/.so next to this file)."""
    global _KERNEL, _KERNEL_TRIED
    if _KERNEL_TRIED:
        return _KERNEL
    _KERNEL_TRIED = True
    try:
        import ctypes
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q3bit_kernel.dll')
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q3bit_kernel.so')
        lib = ctypes.CDLL(path)
        for fn in ('bnb_dequant_3bit', 'bnb_dequant_2bit'):
            getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_longlong, ctypes.c_int]
            getattr(lib, fn).restype = None
        _KERNEL = lib
    except Exception as exc:
        print('  [kernel] not available:', exc)
        _KERNEL = None
    return _KERNEL


# ---------------------------------------------------------------------------
# blockwise quantization: per-block absmax, symmetric levels (2/3/4/8 bit)
# value(code) = code * (absmax/center) - absmax,  center = (2^bits - 1)/2
# ---------------------------------------------------------------------------
def _levels(bits: int) -> int:
    return 1 << bits


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """codes: (nblocks, block) uint8 with block % 8 == 0; returns (nblocks, bytes_per_block)."""
    nb, block = codes.shape
    if bits == 2:
        c = codes.view(nb, block // 4, 4).to(torch.int32)
        v = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
        return v.to(torch.uint8).reshape(nb, -1)
    if bits == 3:
        c = codes.view(nb, block // 8, 8).to(torch.int32)
        v = (c[..., 0] | (c[..., 1] << 3) | (c[..., 2] << 6) | (c[..., 3] << 9)
             | (c[..., 4] << 12) | (c[..., 5] << 15) | (c[..., 6] << 18) | (c[..., 7] << 21))
        b0 = (v & 0xFF).to(torch.uint8)
        b1 = ((v >> 8) & 0xFF).to(torch.uint8)
        b2 = ((v >> 16) & 0xFF).to(torch.uint8)
        return torch.stack([b0, b1, b2], dim=-1).reshape(nb, -1)
    raise ValueError("only 2/3-bit packing implemented")


def unpack_codes(packed: torch.Tensor, bits: int, block: int) -> torch.Tensor:
    nb = packed.shape[0]
    if bits == 2:
        v = packed.reshape(nb, -1, 1).to(torch.int32)
        vals = [(v[..., 0] >> s) & 0x03 for s in (0, 2, 4, 6)]
        return torch.stack(vals, dim=-1).reshape(nb, block).to(torch.uint8)
    if bits == 3:
        p = packed.reshape(nb, -1, 3).to(torch.int32)
        v = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
        vals = [(v >> (3 * i)) & 0x07 for i in range(8)]
        return torch.stack(vals, dim=-1).reshape(nb, block).to(torch.uint8)
    raise ValueError("only 2/3-bit packing implemented")


def quantize_blocks(w: torch.Tensor, bits: int, block: int = 64):
    """Returns packed, absmax, orig_shape, n_elems, block."""
    x = w.detach().float().contiguous()
    shape = x.shape
    flat = x.reshape(-1)
    n = flat.numel()
    pad = (block - n % block) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    nb = flat.numel() // block
    blk = flat.view(nb, block)
    absmax = blk.abs().amax(dim=1)
    if bits >= 8:
        # 8-bit path: reuse the same formula (center=127.5) as a reference
        pass
    center = (_levels(bits) - 1) / 2.0
    scale = (absmax / center).clamp_min(1e-12)
    q = torch.round(blk / scale.unsqueeze(1) + center).clamp(0, _levels(bits) - 1)
    packed = pack_codes(q.to(torch.uint8), bits)
    return packed, absmax, shape, n


def dequantize_blocks(packed, absmax, shape, n, bits, block=64):
    codes = unpack_codes(packed, bits, block)
    center = (_levels(bits) - 1) / 2.0
    scale = absmax / center
    vals = codes.float() * scale.unsqueeze(1) - absmax.unsqueeze(1)
    return vals.reshape(-1)[:n].reshape(shape)


def quantize_blocks_8bit(w: torch.Tensor, block: int = 64):
    x = w.detach().float().contiguous()
    shape = x.shape
    flat = x.reshape(-1)
    n = flat.numel()
    pad = (block - n % block) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    nb = flat.numel() // block
    blk = flat.view(nb, block)
    absmax = blk.abs().amax(dim=1)
    scale = (absmax / 127.5).clamp_min(1e-12)
    q = torch.round(blk / scale.unsqueeze(1) + 127.5).clamp(0, 255).to(torch.uint8)
    return q, absmax, shape, n


# ---------------------------------------------------------------------------
class QuantLinear(nn.Module):
    """Frozen base weight stored as 2/3-bit blocks + trainable LoRA adapters."""

    def __init__(self, base: nn.Linear, bits: int, block: int = 64,
                 r: int = 8, alpha: int = 16, keep_fp32: bool = False,
                 use_kernel: bool = False):
        super().__init__()
        self.bits = bits
        self.use_kernel = bool(use_kernel) and bits in (2, 3)
        self.block = block
        self.in_features = base.in_features
        self.out_features = base.out_features
        w = base.weight.detach()
        if keep_fp32 or bits >= 32:
            self.register_buffer('weight_fp32', w.float().contiguous())
            self.packed = None
            self.absmax = None
            self.shape = w.shape
            self.n = w.numel()
        else:
            packed, absmax, shape, n = quantize_blocks(w, bits, block)
            self.register_buffer('packed', packed)
            self.register_buffer('absmax', absmax)
            self.shape = shape
            self.n = n
            self.weight_fp32 = None
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone().float(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.scale = alpha / r
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def weight(self) -> torch.Tensor:
        if self.weight_fp32 is not None:
            return self.weight_fp32
        if self.use_kernel and _KERNEL is not None:
            out = torch.empty(self.n, dtype=torch.float32)
            fn = _KERNEL.bnb_dequant_3bit if self.bits == 3 else _KERNEL.bnb_dequant_2bit
            fn(self.packed.data_ptr(), self.absmax.data_ptr(), out.data_ptr(),
               self.absmax.numel(), self.block)
            return out.reshape(self.shape).to(self.lora_A.dtype)
        return dequantize_blocks(self.packed, self.absmax, self.shape, self.n,
                                 self.bits, self.block).to(self.lora_A.dtype)

    def forward(self, x):
        y = F.linear(x, self.weight(), self.bias)
        return y + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale


class QuantEmbedding(nn.Module):
    """Frozen 2/3-bit embedding table; lookups dequantize only the selected rows.

    The table is quantized per row with blocks along the hidden dimension, so a lookup of
    N tokens only unpacks N rows instead of the whole vocab (this is what makes quantized
    embeddings actually cheaper at inference/training time).
    """

    def __init__(self, base: nn.Embedding, bits: int, block: int = 64):
        super().__init__()
        self.bits = bits
        self.block = block
        self.num_embeddings = base.num_embeddings
        self.embedding_dim = base.embedding_dim
        w = base.weight.detach().float()
        self.padded_dim = ((self.embedding_dim + block - 1) // block) * block
        if self.padded_dim != self.embedding_dim:
            w = F.pad(w, (0, self.padded_dim - self.embedding_dim))
        packed_chunks, absmax_chunks = [], []
        nb_h = self.padded_dim // block
        center = (_levels(bits) - 1) / 2.0
        for start in range(0, self.num_embeddings, 4096):
            chunk = w[start:start + 4096]
            chunk_n = chunk.shape[0]
            blocks = []
            amaxs = []
            for i in range(nb_h):
                blk = chunk[:, i * block:(i + 1) * block]
                amax = blk.abs().amax(dim=1)
                scale = (amax / center).clamp_min(1e-12)
                q = torch.round(blk / scale.unsqueeze(1) + center).clamp(0, _levels(bits) - 1)
                qc = q.to(torch.uint8)
                if bits == 3:
                    v = (qc[:, 0::8].to(torch.int32) | (qc[:, 1::8].to(torch.int32) << 3)
                         | (qc[:, 2::8].to(torch.int32) << 6) | (qc[:, 3::8].to(torch.int32) << 9)
                         | (qc[:, 4::8].to(torch.int32) << 12) | (qc[:, 5::8].to(torch.int32) << 15)
                         | (qc[:, 6::8].to(torch.int32) << 18) | (qc[:, 7::8].to(torch.int32) << 21))
                    bs = torch.stack([(v & 0xFF).to(torch.uint8), ((v >> 8) & 0xFF).to(torch.uint8),
                                      ((v >> 16) & 0xFF).to(torch.uint8)], dim=-1)
                    blocks.append(bs.reshape(chunk_n, -1))
                else:
                    v = (qc[:, 0::4].to(torch.int32) | (qc[:, 1::4].to(torch.int32) << 2)
                         | (qc[:, 2::4].to(torch.int32) << 4) | (qc[:, 3::4].to(torch.int32) << 6))
                    blocks.append(v.to(torch.uint8))
                amaxs.append(amax)
            packed_chunks.append(torch.cat(blocks, dim=1))   # (chunk_n, nb_h*bytes)
            absmax_chunks.append(torch.stack(amaxs, dim=1))  # (chunk_n, nb_h)
        self.register_buffer('packed', torch.cat(packed_chunks, dim=0))
        self.register_buffer('absmax', torch.cat(absmax_chunks, dim=0))

    def forward(self, ids):
        shape = ids.shape
        idx = ids.reshape(-1)
        packed = self.packed[idx]       # (N, nb_h*bytes)
        absmax = self.absmax[idx]       # (N, nb_h)
        blocks = []
        nb_h = self.padded_dim // self.block
        bytes_per_block = packed.shape[1] // nb_h
        for i in range(nb_h):
            p = packed[:, i * bytes_per_block:(i + 1) * bytes_per_block]
            codes = unpack_codes(p, self.bits, self.block)
            center = (_levels(self.bits) - 1) / 2.0
            scale = absmax[:, i:i + 1] / center
            blocks.append(codes.float() * scale - absmax[:, i:i + 1])
        out = torch.cat(blocks, dim=1)[:, :self.embedding_dim]
        return out.reshape(*shape, self.embedding_dim)


def quantize_embeddings(model: nn.Module, bits: int, block: int = 64) -> int:
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Embedding) and child.num_embeddings > 64:
            setattr(model, name, QuantEmbedding(child, bits, block))
            n += 1
        else:
            n += quantize_embeddings(child, bits, block)
    return n


def quantize_model(model: nn.Module, bits: int, block: int = 64, r: int = 8,
                   keep_fp32: bool = False, use_kernel: bool = False):
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            if name == 'lm_head':
                continue
            setattr(model, name, QuantLinear(child, bits, block, r=r, keep_fp32=keep_fp32,
                                             use_kernel=use_kernel))
            n += 1
        else:
            n += quantize_model(child, bits, block, r, keep_fp32, use_kernel)
    return n


def hidden_from(model, ids):
    out = model(input_ids=ids, output_hidden_states=True)
    if getattr(out, 'last_hidden_state', None) is not None:
        return out.last_hidden_state
    return out.hidden_states[-1]


def model_mem_gb(model: nn.Module) -> float:
    tot = 0
    for p in model.parameters():
        tot += p.numel() * p.element_size()
    seen = set()
    for b in model.buffers():
        tot += b.numel() * b.element_size()
    return tot / (1 << 30)


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1 << 30)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
def real_weight_samples():
    """A few real Qwen3.5-0.8B linear weights for the error table (small + large)."""
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        r'D:\deepsleep\_test\models\qwen3.5-0.8B', dtype=torch.float32,
        low_cpu_mem_usage=True)
    weights = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            weights.append((name, mod.weight.detach().clone()))
            if len(weights) >= 3:
                break
    return weights


def packtest():
    print('=== pack/unpack round-trip (codes must be exact) ===')
    for bits in (2, 3):
        w = torch.randn(4, 64)
        packed, absmax, shape, n = quantize_blocks(w, bits, 64)
        back = dequantize_blocks(packed, absmax, shape, n, bits, 64)
        wq = torch.round(w)  # not expected; just check shape/finite
        print('bits=%d packed=%s bytes/blk dequant finite=%s shape=%s'
              % (bits, tuple(packed.shape), bool(torch.isfinite(back).all()),
                 tuple(back.shape)))
        # code-level round-trip: quantize then unpack must reproduce the code tensor
        codes = unpack_codes(packed, bits, 64)
        print('   codes range', int(codes.min()), int(codes.max()),
              '| packed bytes per 64 elems:', packed.numel() // packed.shape[0])

    print('\n=== quantization error on random N(0,1) (relative L2) ===')
    torch.manual_seed(0)
    w = torch.randn(1 << 20)
    rows = []
    for bits in (2, 3, 8):
        if bits == 8:
            q, absmax, shape, n = quantize_blocks_8bit(w, 64)
            center = 127.5
            scale = absmax / center
            codes = q.view(-1).float()
            back = (codes * scale.repeat_interleave(64)[:n] - absmax.repeat_interleave(64)[:n])
        else:
            packed, absmax, shape, n = quantize_blocks(w, bits, 64)
            back = dequantize_blocks(packed, absmax, shape, n, bits, 64)
        rel = (w - back).norm() / w.norm()
        rows.append((bits, float(rel)))
        mb = (w.numel() * bits / 8) / (1 << 20) if bits < 8 else w.numel() / (1 << 20)
        print('  %d-bit: relL2=%.4f  packed=%.2f MB (plus absmax %.2f MB)'
              % (bits, rel, mb, (w.numel() / 64 * 4) / (1 << 20)))

    print('\n=== quantization error on real Qwen3.5-0.8B weights ===')
    for name, w in real_weight_samples():
        line = '  %-45s shape=%-18s' % (name.split('.')[-2:], tuple(w.shape))
        errs = []
        for bits in (2, 3, 8):
            if bits == 8:
                q, absmax, shape, n = quantize_blocks_8bit(w, 64)
                back = (q.view(-1).float() * (absmax / 127.5).repeat_interleave(64)[:n]
                        - absmax.repeat_interleave(64)[:n]).reshape(w.shape)
            else:
                packed, absmax, shape, n = quantize_blocks(w, bits, 64)
                back = dequantize_blocks(packed, absmax, shape, n, bits, 64)
            errs.append('%d-bit relL2=%.4f' % (bits, float((w - back).norm() / w.norm())))
        print(line + ' | ' + ' | '.join(errs))
    print('PACKTEST DONE')


# ---------------------------------------------------------------------------
def load_model(which: str):
    from transformers import (Qwen3_5ForConditionalGeneration, BertForMaskedLM,
                              AutoModelForCausalLM)
    if which == 'qwen':
        m = Qwen3_5ForConditionalGeneration.from_pretrained(
            r'D:\deepsleep\_test\models\qwen3.5-0.8B', dtype=torch.float32,
            low_cpu_mem_usage=True)
        return m, m.model, getattr(m.config, 'text_config', m.config).hidden_size
    # tiny local model used by the health matrix
    path = r'D:\deepsleep\tiny_models\hf-internal-testing__tiny-random-BertForMaskedLM'
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    m.train()
    return m, m, 32


def train(args):
    import bitsandbytes
    torch.set_num_threads(6)
    if args.kernel:
        load_kernel()
    t0 = time.time()
    outer, model, hidden = load_model(args.model)
    outer.train()
    for p in outer.parameters():
        p.requires_grad_(False)
    n_emb = 0
    if args.quant_emb and args.bits < 32:
        if hasattr(outer, 'lm_head'):
            outer.lm_head = None  # not used by this hidden-state objective; frees the tied fp32 table
        n_emb = quantize_embeddings(model, args.bits, args.block)
    n_lora = quantize_model(model, args.bits, args.block, args.lora_r,
                            keep_fp32=(args.bits >= 32), use_kernel=args.kernel)
    gc.collect()
    trainable = [p for p in outer.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
    print('loaded %.1fs | quantized %d linears + %d embeddings to %s-bit | trainable %.2fM | weight-mem %.2f GB | rss %.2f GB'
          % (time.time() - t0, n_lora, n_emb, args.bits, sum(p.numel() for p in trainable) / 1e6,
             model_mem_gb(outer), rss_gb()))

    torch.manual_seed(0)
    ids = torch.randint(1, 1000, (1, args.seq))
    target = torch.randn(1, args.seq, hidden) * 0.5
    times, losses = [], []
    for step in range(args.steps + 1):
        t = time.perf_counter()
        h = hidden_from(model, ids)
        t_fwd = time.perf_counter()
        loss = F.mse_loss(h, target)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        t_end = time.perf_counter()
        dt = t_end - t
        if os.environ.get('BNB_Q3_PROFILE'):
            print('    [phase] fwd %.0f | rest %.0f ms' % ((t_fwd - t) * 1e3, (t_end - t_fwd) * 1e3))
        if step:
            times.append(dt)
            losses.append(float(loss.detach()))
        if step == 0 or step == args.steps:
            print('  step %d loss %.6f rss %.2f GB' % (step, float(loss.detach()), rss_gb()))
    tab = {'model': args.model, 'bits': args.bits, 'block': args.block, 'seq': args.seq,
           'steps': args.steps, 'lora_r': args.lora_r,
           'step_ms_median': round(1000 * sorted(times)[len(times) // 2], 1),
           'loss_first': round(losses[0], 6), 'loss_last': round(losses[-1], 6),
           'weight_mem_gb': round(model_mem_gb(outer), 3), 'rss_gb': round(rss_gb(), 2)}
    print('RESULT ' + str(tab))
    print('TRAIN DONE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['packtest', 'train'], required=True)
    ap.add_argument('--model', choices=['qwen', 'tiny'], default='tiny')
    ap.add_argument('--bits', type=int, default=3)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--quant-emb', action='store_true')
    ap.add_argument('--kernel', action='store_true')
    args = ap.parse_args()
    if args.mode == 'packtest':
        packtest()
    else:
        train(args)


if __name__ == '__main__':
    main()
