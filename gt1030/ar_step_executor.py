"""P1a executor prototype: a real AR-step-shaped training step with the GEMMs on the iGPU.

The schedule benchmark said the GEMM phase alone is 0.515 s (1.15x the GT 1030 reference). This script
turns that into a measured *whole step*: a transformer with the AR model's shape (d512, 10 layers,
8/4 GQA heads, SwiGLU hidden 1368, vocab 1024, 2048 tokens/step) whose linear layers run on the
iGPU via fp16 CLBlast, while everything else (RMSNorm, SiLU/gate, softmax, loss, AdamW) stays on the
CPU. That is the phase-separation architecture from report 7.118.8.

Attention's QK^T/P@V stay on the CPU on purpose: their FLOPs are <0.5% of the step, and the AR
model's attention path is small (T=256/head). What we measure is the part that matters.

Two runs, same seed/init/data:
  CPU baseline (plain torch fp32 linears)  -> loss curve + step time reference
  iGPU offload                            -> same loss (parity check) + step time
If the losses agree, the offload preserves the training semantics at this scale; if the step time is
within the projection, the executor is real.
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

torch.set_num_threads(6)
torch.set_grad_enabled(True)
torch.manual_seed(0)

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ONE, ZERO = 0x3C00, 0x0000
LAYOUT_COL, TRANSPOSE_NO = 0, 111

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))

STATS = {'h2d': 0.0, 'gemm': 0.0, 'd2h': 0.0, 'calls': 0}


def _up(np_fp16):
    t0 = time.perf_counter()
    buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_fp16)
    queue.finish()
    STATS['h2d'] += time.perf_counter() - t0
    return buf


def _down(buf, shape, dtype):
    t0 = time.perf_counter()
    out = np.empty(shape, dtype=dtype)
    cl.enqueue_copy(queue, out, buf)
    queue.finish()
    STATS['d2h'] += time.perf_counter() - t0
    return out


def gemm_nn(a16, b16, m, n, k):
    """C_row(m,n) = A_row(m,k) @ B_row(k,n), both fp16 host arrays -> fp16 host array."""
    ab = _up(a16)
    bb = _up(b16)
    cb = cl.Buffer(ctx, mf.READ_WRITE, m * n * 2)
    t0 = time.perf_counter()
    rc = bl.CLBlastHgemm(
        LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ONE,
        ctypes.c_void_p(int(bb.int_ptr)), 0, n,
        ctypes.c_void_p(int(ab.int_ptr)), 0, k, ZERO,
        ctypes.c_void_p(int(cb.int_ptr)), 0, n,
        ctypes.byref(QP), None)
    queue.finish()
    STATS['gemm'] += time.perf_counter() - t0
    STATS['calls'] += 1
    if rc != 0:
        raise RuntimeError('Hgemm rc=%d shape %dx%dx%d' % (rc, m, n, k))
    return _down(cb, (m, n), np.float16)


class IgpuLinear(nn.Module):
    """A linear layer whose GEMMs run on the iGPU: y = x @ W^T with W (out,in) torch fp32 master."""

    def __init__(self, out_features: int, in_features: int, bias: bool = False):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = None
        self._dev = None   # (W (in,out) fp16, Wt (out,in) fp16) as numpy arrays on host, uploaded per call

    def refresh(self):
        w = self.weight.detach().numpy().astype(np.float16)      # (out,in)
        self._wt = np.ascontiguousarray(w)                        # (out,in) = W^T for dX
        self._w = np.ascontiguousarray(w.T)                       # (in,out)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._dev is None:
            self.refresh()
        return _IgpuLinearFn.apply(x, self)


class _IgpuLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mod: IgpuLinear):
        orig_shape = x.shape
        x2 = x.reshape(-1, mod.in_features).detach().float().numpy().astype(np.float16)
        m = x2.shape[0]
        out16 = gemm_nn(np.ascontiguousarray(x2), mod._w, m, mod.out_features, mod.in_features)
        ctx.x16 = x2
        ctx.mod = mod
        ctx.m = m
        ctx.orig_shape = orig_shape
        return torch.from_numpy(out16.astype(np.float32)).reshape(*orig_shape[:-1], mod.out_features)

    @staticmethod
    def backward(ctx, grad_out):
        g = grad_out.reshape(-1, ctx.mod.out_features).detach().float().numpy().astype(np.float16)
        g = np.ascontiguousarray(g)
        mod = ctx.mod
        # dX = dY @ W  (dY (m,out) @ W (out,in) where W = mod._wt (out,in))
        dx16 = gemm_nn(g, mod._wt, ctx.m, mod.in_features, mod.out_features)
        # dW = X^T @ dY : (in,m) @ (m,out)
        xt = np.ascontiguousarray(ctx.x16.T)
        dw16 = gemm_nn(xt, g, mod.in_features, mod.out_features, ctx.m)   # (in,out)
        dw = torch.from_numpy(np.ascontiguousarray(dw16.T).astype(np.float32))  # -> (out,in)
        if mod.weight.grad is None:
            mod.weight.grad = dw
        else:
            mod.weight.grad.add_(dw)
        return torch.from_numpy(dx16.astype(np.float32)).reshape(ctx.orig_shape), None


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Block(nn.Module):
    def __init__(self, d, ff, n_head, n_kv):
        super().__init__()
        self.n_head, self.n_kv, self.dh = n_head, n_kv, d // n_head
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.qkv = IgpuLinear(d + 2 * n_kv * self.dh, d)
        self.o = IgpuLinear(d, d)
        self.w1 = IgpuLinear(ff, d)     # SwiGLU gate
        self.w3 = IgpuLinear(ff, d)     # SwiGLU up
        self.w2 = IgpuLinear(d, ff)     # SwiGLU down
        self.ls1 = nn.Parameter(torch.full((d,), 1e-2))
        self.ls2 = nn.Parameter(torch.full((d,), 1e-2))

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, self.n_head + 2 * self.n_kv, self.dh).transpose(1, 2)
        q, k, v = qkv.split([self.n_head, self.n_kv, self.n_kv], dim=1)
        k = k.repeat_interleave(self.n_head // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv, dim=1)
        s = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        s = s.masked_fill(~mask, float('-inf'))
        p = torch.softmax(s, dim=-1)
        a = (p @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.ls1 * self.o(a)
        h = self.ln2(x)
        gate = F.silu(self.w1(h))
        up = self.w3(h)
        x = x + self.ls2 * self.w2(gate * up)
        return x


class MiniAR(nn.Module):
    def __init__(self, d=512, ff=1368, nl=10, nh=8, nkv=4, vocab=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, ff, nh, nkv) for _ in range(nl)])
        self.norm = RMSNorm(d)
        self.head = IgpuLinear(vocab, d)
        self.d, self.vocab = d, vocab

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def igpu_linears(self):
        yield self.head
        for b in self.blocks:
            yield b.qkv
            yield b.o
            yield b.w1
            yield b.w3
            yield b.w2


def cpu_ref_model(d=512, ff=1368, nl=10, nh=8, nkv=4, vocab=1024):
    """Same math with plain torch linears (fp32). Weight names are mapped by hand."""
    dh = d // nh
    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
            self.qi = nn.Linear(d, d, bias=False)
            self.ki = nn.Linear(d, nkv * (d // nh), bias=False)
            self.vi = nn.Linear(d, nkv * (d // nh), bias=False)
            self.o = nn.Linear(d, d, bias=False)
            self.w1 = nn.Linear(d, ff, bias=False)
            self.w3 = nn.Linear(d, ff, bias=False)
            self.w2 = nn.Linear(ff, d, bias=False)
            self.ls1 = nn.Parameter(torch.full((d,), 1e-2))
            self.ls2 = nn.Parameter(torch.full((d,), 1e-2))

        def forward(self, x):
            B_, T, D = x.shape
            h = self.ln1(x)
            q = self.qi(h).view(B_, T, nh, dh).transpose(1, 2)
            k = self.ki(h).view(B_, T, nkv, dh).transpose(1, 2).repeat_interleave(nh // nkv, dim=1)
            v = self.vi(h).view(B_, T, nkv, dh).transpose(1, 2).repeat_interleave(nh // nkv, dim=1)
            s = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            s = s.masked_fill(~mask, float('-inf'))
            a = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B_, T, D)
            x = x + self.ls1 * self.o(a)
            h = self.ln2(x)
            x = x + self.ls2 * self.w2(F.silu(self.w1(h)) * self.w3(h))
            return x

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.blocks = nn.ModuleList([B() for _ in range(nl)])
            self.norm = RMSNorm(d)
            self.head = nn.Linear(d, vocab, bias=False)

        def forward(self, ids):
            x = self.emb(ids)
            for b in self.blocks:
                x = b(x)
            return self.head(self.norm(x))
    return M()


def sync_ref_weights(ref: nn.Module, ig: MiniAR):
    """Copy the reference model's weights INTO the iGPU model (same starting point for parity)."""
    with torch.no_grad():
        ig.emb.weight.copy_(ref.emb.weight)
        ig.norm.w.copy_(ref.norm.w)
        ig.head.weight.copy_(ref.head.weight)
        for bi, b in enumerate(ig.blocks):
            rb = ref.blocks[bi]
            b.ln1.w.copy_(rb.ln1.w)
            b.ln2.w.copy_(rb.ln2.w)
            w = rb.qkv_weight if hasattr(rb, 'qkv_weight') else None
            # rebuild the fused qkv by concatenating the reference's separate projections
            d_out = b.qkv.weight.shape[0]
            d_in = b.qkv.weight.shape[1]
            kv = (d_out - d_in) // 2
            fused = torch.cat([rb.qi.weight, rb.ki.weight, rb.vi.weight], dim=0)
            b.qkv.weight.copy_(fused)
            b.o.weight.copy_(rb.o.weight)
            b.w1.weight.copy_(rb.w1.weight)
            b.w3.weight.copy_(rb.w3.weight)
            b.w2.weight.copy_(rb.w2.weight)
            b.ls1.copy_(rb.ls1)
            b.ls2.copy_(rb.ls2)


def loss_of(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def main() -> int:
    import os
    small = os.environ.get('BNB_EXEC_SMALL') == '1'
    if small:
        D, FF, NL, NH, NKV, VOCAB, B, T, STEPS = 64, 128, 2, 2, 1, 64, 2, 32, 3
    else:
        D, FF, NL, NH, NKV, VOCAB, B, T, STEPS = 512, 1368, 10, 8, 4, 1024, 8, 256, 8
    print('iGPU: %s (%d CU) | d=%d ff=%d layers=%d heads=%d/%d B=%d T=%d steps=%d'
          % (ctx.devices[0].name, ctx.devices[0].max_compute_units, D, FF, NL, NH, NKV, B, T, STEPS))
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (B, T + 1))
    inp, tgt = ids[:, :-1], ids[:, 1:]

    # --- CPU reference: loss curve + step time ---
    ref = cpu_ref_model(D, FF, NL, NH, NKV, VOCAB)
    init_state = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    for p in ref.parameters():
        p.grad = None
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=5e-4)
    ref_losses = []
    ref_step_times = []
    for _ in range(STEPS):
        t0 = time.perf_counter()
        opt_ref.zero_grad()
        l = loss_of(ref(inp), tgt)
        l.backward()
        ref_losses.append(float(l))
        opt_ref.step()
        ref_step_times.append(time.perf_counter() - t0)
    print('CPU reference losses:', ' '.join('%.4f' % v for v in ref_losses))

    # --- iGPU model: start from the SAME initial weights as the reference ---
    ref0 = cpu_ref_model(D, FF, NL, NH, NKV, VOCAB)
    ref0.load_state_dict(init_state)
    ig = MiniAR(D, FF, NL, NH, NKV, VOCAB)
    sync_ref_weights(ref0, ig)
    for m in (ref0, ig):
        for p in m.parameters():
            p.grad = None
    opt = torch.optim.AdamW(ig.parameters(), lr=5e-4)
    losses = []
    step_times = []
    for s in range(STEPS):
        for k in STATS:
            STATS[k] = 0.0 if k != 'calls' else 0
        for lin in ig.igpu_linears():      # sync fp32 masters -> fp16 device copies (once per step)
            lin.refresh()
        t0 = time.perf_counter()
        opt.zero_grad()
        l = loss_of(ig(inp), tgt)
        l.backward()
        losses.append(float(l))
        opt.step()
        step_times.append(time.perf_counter() - t0)
    print('iGPU offload losses:', ' '.join('%.4f' % v for v in losses))
    print('max |loss_iGPU - loss_CPU|: %.3e' % max(abs(a - b) for a, b in zip(losses, ref_losses)))

    med = statistics.median(step_times[2:])
    flops = 364e9
    print()
    print('whole-step time (median of %d, 2 warmup): %.3f s' % (STEPS - 2, med))
    print('  breakdown last step: H2D %.1f ms  GEMM %.1f ms  D2H %.1f ms  (%d GEMM calls)'
          % (STATS['h2d'] * 1e3, STATS['gemm'] * 1e3, STATS['d2h'] * 1e3, STATS['calls']))
    print('effective throughput: %.0f GFLOPS  (GT 1030 reference 0.594 s -> fraction %.2fx)'
          % (flops / med / 1e9, 0.594 / med))
    print('CPU baseline last loss %.4f vs reference %.4f' % (losses[-1], ref_losses[-1]))
    ref_med = statistics.median(ref_step_times[2:])
    print('CPU reference step time: %.3f s (iGPU %.3f s, speedup %.2fx)'
          % (ref_med, med, ref_med / med))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
