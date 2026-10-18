"""Train a real HuggingFace model on the iGPU: frozen base + LoRA adapters.

iGPU path: every frozen nn.Linear's forward (y = x W^T) and backward (dX = dY W) runs as a CLBlast
**Sgemm in fp32** with the weights resident on the device once per layer. fp32 accumulation is what
fixes the precision: the earlier fp16 Hgemm path diverged (step-9 loss 7.32 vs CPU 0.083) because of
half-precision accumulation over K=2048/5632, while a hand-written fp16-storage/fp32-accumulate kernel
matched the CPU (final losses 0.0079 vs 0.0159) but ran at only ~42 GFLOPS. fp32 Sgemm is exact and
CLBlast-optimised; the model's fp32 weights are 5.2 GB on the device (1.3B params), which fits in the
15.4 GB shared memory alongside the CPU copy.

Two passes over ONE model instance (same init, same data):
  * CPU pass: the same wrapper, but the frozen GEMM is a torch CPU matmul (CPU_MODE);
  * iGPU pass: the frozen GEMM runs on the iGPU.
This gives a direct precision/speed comparison.

Usage:
  python gt1030/igpu_train_lora.py --model <dir> --steps 500 --cpu-steps 30 \
      [--batch 1] [--tokens 64] [--lr 1e-3] [--log-every 25]
"""
from __future__ import annotations

import argparse
import ctypes
import math
import statistics
import sys
import time

import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO, YES = 101, 111, 112

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))

CPU_MODE = False
STATS = {'gemm': 0.0, 'calls': 0}
_POOL = {}


def _buf(role, nbytes):
    key = (role, nbytes)
    b = _POOL.get(key)
    if b is None:
        b = cl.Buffer(ctx, mf.READ_WRITE, nbytes)
        _POOL[key] = b
    return b


def sgemm(a_dev, b_dev, c_dev, m, n, k, lda, ldb, ldc, trans_b):
    """C(m,n) = A @ op(B) in fp32, row-major. trans_b: B stored (n,k) instead of (k,n)."""
    rc = bl.CLBlastSgemm(ROW, NO, YES if trans_b else NO, m, n, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(a_dev.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(b_dev.int_ptr)), 0, ldb, ctypes.c_float(0.0),
                         ctypes.c_void_p(int(c_dev.int_ptr)), 0, ldc,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Sgemm rc=%d %dx%dx%d' % (rc, m, n, k))


def base_forward(x_np, w_dev, m, n, k):
    ad = _buf('a32', m * k * 4)
    cl.enqueue_copy(queue, ad, np.ascontiguousarray(x_np, dtype=np.float32))
    cd = _buf('c32', m * n * 4)
    sgemm(ad, w_dev, cd, m, n, k, k, k, n, True)     # y = x @ W^T, W stored (n,k)
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, cd)
    queue.finish()
    STATS['calls'] += 1
    return out


def base_backward(g_np, w_dev, m, n, k):
    ad = _buf('a32', m * n * 4)
    cl.enqueue_copy(queue, ad, np.ascontiguousarray(g_np, dtype=np.float32))
    cd = _buf('c32', m * k * 4)
    sgemm(ad, w_dev, cd, m, k, n, n, k, k, False)    # dX = g @ W, W stored (n,k)
    out = np.empty((m, k), dtype=np.float32)
    cl.enqueue_copy(queue, out, cd)
    queue.finish()
    STATS['calls'] += 1
    return out


class FrozenBaseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp32, w_dev):
        orig = x.shape
        k = orig[-1]
        n = w_fp32.shape[0]
        ctx.orig = orig
        ctx.w_fp32 = w_fp32
        ctx.w_dev = w_dev
        if CPU_MODE:
            return (x.reshape(-1, k) @ w_fp32.t()).reshape(*orig[:-1], n)
        x_np = x.detach().reshape(-1, k).float().numpy()
        m = x_np.shape[0]
        t0 = time.perf_counter()
        out = base_forward(x_np, w_dev, m, n, k)
        STATS['gemm'] += time.perf_counter() - t0
        return torch.from_numpy(out).reshape(*orig[:-1], n)

    @staticmethod
    def backward(ctx, gy):
        orig = ctx.orig
        k = orig[-1]
        n = gy.shape[-1]
        if CPU_MODE:
            return (gy.reshape(-1, n) @ ctx.w_fp32).reshape(*orig), None, None
        g_np = gy.detach().reshape(-1, n).float().numpy()
        m = g_np.shape[0]
        out = base_backward(g_np, ctx.w_dev, m, n, k)   # dX = g @ W
        return torch.from_numpy(out).reshape(*orig[:-1], k), None, None


class IgpuLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.base_weight = base.weight.detach()                 # shared reference (frozen)
        self.base_bias = base.bias.detach() if base.bias is not None else None
        self.use_bias = base.bias is not None
        self.w_dev = None
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def ensure_dev(self):
        if self.w_dev is None and not CPU_MODE:
            w = np.ascontiguousarray(self.base_weight.detach().numpy(), dtype=np.float32)
            self.w_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        self.ensure_dev()
        y = FrozenBaseFn.apply(x, self.base_weight, self.w_dev)
        if self.use_bias:
            y = y + self.base_bias
        return y + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale


def patch_model(model: nn.Module, r: int) -> int:
    count = 0

    def walk(mod):
        nonlocal count
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, name, IgpuLoRALinear(child, r))
                count += 1
            else:
                walk(child)
    walk(model)
    return count


def train(model, args, steps, cpu_mode, lora_state):
    global CPU_MODE
    CPU_MODE = cpu_mode
    if lora_state is not None:
        model.load_state_dict(lora_state, strict=False)
    gen = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, 2000, (args.batch, args.tokens + 1), generator=gen)
    inp, tgt = ids[:, :-1], ids[:, 1:]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    losses, times = [], []
    for step in range(steps):
        if not cpu_mode:
            STATS['gemm'] = 0.0
            STATS['calls'] = 0
        t0 = time.perf_counter()
        out = model(inp, labels=tgt, use_cache=False)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == steps - 1:
            msg = '  [%s] step %3d  loss %.4f  step %.3f s' % (
                'cpu' if cpu_mode else 'igpu', step, losses[-1], times[-1])
            if not cpu_mode:
                msg += '  (calls %d, gemm %.0f ms)' % (STATS['calls'], STATS['gemm'] * 1e3)
            print(msg, flush=True)
    CPU_MODE = False
    return losses, times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--cpu-steps', type=int, default=30)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--log-every', type=int, default=25)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    torch.set_num_threads(6)
    print('device:', ctx.devices[0].name)
    print('loading', args.model)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = patch_model(model, args.lora_r)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('loaded %.1f s | %d layers patched | trainable %.2f M'
          % (time.perf_counter() - t0, n, trainable / 1e6))
    lora_init = {k: v.detach().clone() for k, v in model.state_dict().items() if 'lora_' in k}

    cpu_losses = []
    if args.cpu_steps > 0:
        cpu_losses, _ = train(model, args, args.cpu_steps, cpu_mode=True, lora_state=lora_init)
        model.load_state_dict(lora_init, strict=False)

    if args.steps == 0:
        print('CPU-only run finished')
        return 0

    t0 = time.perf_counter()
    losses, times = train(model, args, args.steps, cpu_mode=False, lora_state=None)
    total = time.perf_counter() - t0
    med = statistics.median(times[5:]) if len(times) > 6 else statistics.median(times)
    print()
    print('iGPU training: %d steps in %.1f s (median %.3f s/step)' % (args.steps, total, med))
    print('loss: first %.4f -> last %.4f (min %.4f)' % (losses[0], losses[-1], min(losses)))
    finite = all(math.isfinite(v) for v in losses)
    print('normal + learning:', 'YES' if (finite and losses[-1] < losses[0]) else 'NO')
    if cpu_losses:
        m = min(len(cpu_losses), len(losses))
        diff = max(abs(a - b) for a, b in zip(cpu_losses[:m], losses[:m]))
        print('precision vs CPU baseline: max |loss diff| over %d steps = %.5f' % (m, diff))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
