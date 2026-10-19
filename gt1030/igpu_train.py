"""Universal iGPU training kit: train ANY HuggingFace model with the iGPU doing the Linear GEMMs.

Backends:
  hybrid  (default, universal): every nn.Linear is replaced by an iGPU-backed module; frozen base + LoRA
          adapters, or full fine-tuning (--mode full) for models small enough.
  resident (llama-family only, in progress): device-resident engine from gt1030/llama_resident.py.

The point of this file is generality: it never assumes a specific architecture -- it works on whatever
`AutoModelForCausalLM` produces, because it only patches torch modules. It also auto-downloads a model
from the Hub through hf-mirror.com when a local path is not given.

Modes:
  --check N   run N steps and print a health report (loss curve, grad norms, finiteness, timing)
  --steps N   run N training steps (default synthetic fixed batch; --data for a text file)

Examples:
  python gt1030/igpu_train.py --model D:\\deepsleep\\tiny_models\\sshleifer__tiny-gpt2 --steps 20
  python gt1030/igpu_train.py --model Qwen/Qwen2-0.5B --steps 5 --mode lora
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import math
import os
import statistics
import sys
import time
import urllib.request

import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

MIRROR = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
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

STATS = {'calls': 0, 'gemm_s': 0.0}
_POOL: dict = {}
_ROLE = [0]


def _buf(nbytes, role=None):
    if role is None:
        _ROLE[0] += 1
        role = _ROLE[0]
    key = (role, nbytes)
    b = _POOL.get(key)
    if b is None:
        b = cl.Buffer(ctx, mf.READ_WRITE, nbytes)
        _POOL[key] = b
    return b


def sgemm(a, b, c, m, n, k, ta=False, tb=False):
    lda = m if ta else k
    ldb = k if tb else n
    rc = bl.CLBlastSgemm(ROW, YES if ta else NO, YES if tb else NO, m, n, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(a.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(b.int_ptr)), 0, ldb, ctypes.c_float(0.0),
                         ctypes.c_void_p(int(c.int_ptr)), 0, n,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Sgemm rc=%d %dx%dx%d' % (rc, m, n, k))


class IgpuFn(torch.autograd.Function):
    """y = x @ W^T (layout 0, nn.Linear) or y = x @ W (layout 1, GPT2 Conv1D) on the iGPU.

    Backward returns dX and, when the weight is trained, dW in the weight's own layout.
    """

    @staticmethod
    def forward(ctx, x, w, w_dev, want_dw, layout):
        orig = x.shape
        k = orig[-1]
        n = w.shape[1] if layout else w.shape[0]
        x2 = np.ascontiguousarray(x.detach().reshape(-1, k).float().numpy(), dtype=np.float32)
        m = x2.shape[0]
        ad = _buf(m * k * 4, 'actA')
        cl.enqueue_copy(queue, ad, x2)
        cd = _buf(m * n * 4, 'outC')
        t0 = time.perf_counter()
        sgemm(ad, w_dev, cd, m, n, k, tb=(layout == 0))
        out = np.empty(tuple(orig[:-1]) + (n,), dtype=np.float32)
        cl.enqueue_copy(queue, out, cd)
        queue.finish()
        STATS['calls'] += 1
        STATS['gemm_s'] += time.perf_counter() - t0
        ctx.x2, ctx.w_dev, ctx.orig, ctx.k, ctx.want_dw = x2, w_dev, orig, k, want_dw
        ctx.layout = layout
        return torch.from_numpy(out)

    @staticmethod
    def backward(ctx, gy):
        orig, k, want_dw, layout = ctx.orig, ctx.k, ctx.want_dw, ctx.layout
        n = gy.shape[-1]
        g = np.ascontiguousarray(gy.detach().reshape(-1, n).float().numpy(), dtype=np.float32)
        m = g.shape[0]
        ag = _buf(m * n * 4, 'gradA')
        cl.enqueue_copy(queue, ag, g)
        cdx = _buf(m * k * 4, 'dxC')
        sgemm(ag, ctx.w_dev, cdx, m, k, n, tb=(layout == 1))   # dX
        dx = np.empty(tuple(orig[:-1]) + (k,), dtype=np.float32)
        cl.enqueue_copy(queue, dx, cdx)
        dW = None
        if want_dw:
            xd = _buf(m * k * 4, 'xBuf')
            cl.enqueue_copy(queue, xd, ctx.x2)
            cdw = _buf(n * k * 4, 'dwC')
            if layout == 0:
                sgemm(ag, xd, cdw, n, k, m, ta=True)           # dW = g^T x  -> (out,in)
            else:
                sgemm(xd, ag, cdw, k, n, m, ta=True)           # dW = x^T g  -> (in,out)
            dwn = np.empty((n, k) if layout == 0 else (k, n), dtype=np.float32)
            cl.enqueue_copy(queue, dwn, cdw)
            dW = torch.from_numpy(dwn)
        queue.finish()
        return torch.from_numpy(dx), dW, None, None, None


def _as_dev(arr, role):
    """Upload an activation array for a one-off use (pooled by size)."""
    d = _buf(arr.size * 4)
    cl.enqueue_copy(queue, d, np.ascontiguousarray(arr))
    return d


class IgpuConv1D(nn.Module):
    """Wrapper for transformers Conv1D modules (weight stored (in,out), y = x @ W)."""

    def __init__(self, base):
        super().__init__()
        self.legacy = base
        self.weight = nn.Parameter(base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone()) if getattr(base, 'bias', None) is not None else None
        self.in_features = base.weight.shape[0]
        self.out_features = base.weight.shape[1]
        self.w_dev = None

    def refresh(self):
        w = np.ascontiguousarray(self.weight.detach().float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, self.weight, self.w_dev, self.weight.requires_grad, 1)
        if self.bias is not None:
            y = y + self.bias
        return y


class IgpuLinear(nn.Module):
    """Full fine-tuning wrapper: trainable weight, iGPU GEMM forward/backward (dX and dW)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.w_dev = None

    def refresh(self):
        w = np.ascontiguousarray(self.weight.detach().float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, self.weight, self.w_dev, self.weight.requires_grad, 0)
        if self.bias is not None:
            y = y + self.bias
        return y


class IgpuLoRALinear(nn.Module):
    """Frozen base on the iGPU + trainable LoRA on the CPU."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.base_weight = base.weight.detach()
        self.base_bias = base.bias.detach() if base.bias is not None else None
        self.use_bias = base.bias is not None
        self.w_dev = None
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def refresh(self):
        w = np.ascontiguousarray(self.base_weight.float().numpy())
        self.w_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, self.base_weight, self.w_dev, False, 0)
        if self.use_bias:
            y = y + self.base_bias
        return y + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale


def patch_model(model: nn.Module, mode: str, r: int, skip_lm_head: bool) -> int:
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:
        Conv1D = ()
    count = 0

    def walk(mod, prefix=''):
        nonlocal count
        for name, child in list(mod.named_children()):
            full = prefix + name
            if skip_lm_head and full in ('lm_head',):
                continue
            if isinstance(child, nn.Linear):
                new = IgpuLinear(child) if mode == 'full' else IgpuLoRALinear(child, r)
                setattr(mod, name, new)
                count += 1
            elif Conv1D and isinstance(child, Conv1D):
                new = IgpuConv1D(child)
                setattr(mod, name, new)
                count += 1
            else:
                walk(child, full + '.')
    walk(model)
    return count


def resolve_model(spec: str) -> str:
    """Local path or download from the Hub mirror into D:\\deepsleep\\hub_models."""
    if os.path.isdir(spec):
        return spec
    local = os.path.join(r'D:\deepsleep\hub_models', spec.replace('/', '__'))
    if os.path.isdir(local) and os.path.exists(os.path.join(local, 'config.json')):
        return local
    os.makedirs(local, exist_ok=True)
    files = ['config.json', 'generation_config.json', 'model.safetensors',
             'model.safetensors.index.json', 'pytorch_model.bin', 'tokenizer.json',
             'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
             'tokenizer.model']
    for f in files:
        dst = os.path.join(local, f)
        if os.path.exists(dst):
            continue
        url = '%s/%s/resolve/main/%s' % (MIRROR, spec, f)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp, open(dst, 'wb') as out:
                    out.write(resp.read())
                break
            except Exception:
                if os.path.exists(dst):
                    os.remove(dst)
                time.sleep(2)
    return local


def make_data(args, model, tokenizer=None):
    """Synthetic fixed batch (default) or random windows from --data text."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    vocab = getattr(cfg, 'vocab_size', None) or getattr(getattr(cfg, 'text_config', cfg), 'vocab_size', 32000)
    if not args.data:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, min(vocab, 4000), (args.batch, args.tokens + 1), generator=g)
        return ids[:, :-1], ids[:, 1:]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    text = open(args.data, encoding='utf-8', errors='replace').read()
    toks = tok(text)['input_ids']
    if len(toks) < args.tokens + 2:
        raise SystemExit('data file too short after tokenisation: %d tokens' % len(toks))
    g = np.random.default_rng(1234)
    win = args.tokens + 1
    batch = []
    for _ in range(args.batch):
        s = int(g.integers(0, len(toks) - win))
        batch.append(toks[s:s + win])
    ids = torch.tensor(batch, dtype=torch.long)
    return ids[:, :-1], ids[:, 1:]


def train(args) -> int:
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(6)
    path = resolve_model(args.model)
    args.model_path = path
    print('model dir:', path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                 local_files_only=True)
    model.eval()
    if args.mode == 'full':
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    skip_lm = bool(getattr(model.config, 'tie_word_embeddings', False))
    total = sum(p.numel() for p in model.parameters())      # before patching: base weights leave Parameters
    inp, tgt = make_data(args, model)
    # CPU reference on the same batch, before patching: the iGPU first-step loss must match it
    with torch.no_grad():
        try:
            base_loss = float(model(inp, labels=tgt, use_cache=False).loss)
        except Exception:
            base_loss = float('nan')
    n = patch_model(model, args.mode, args.lora_r, skip_lm)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('architecture: %s | params %.2f M | trainable %.2f M (%.1f%%) | iGPU linears %d%s'
          % (model.config.architectures[0] if getattr(model.config, 'architectures', None) else '?',
             total / 1e6, trainable / 1e6, 100.0 * trainable / max(1, total), n,
             ' (lm_head kept on CPU: tied embeddings)' if skip_lm else ''))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    losses, times = [], []
    ok = True
    t_start = time.perf_counter()
    for step in range(args.steps):
        for k in STATS:
            STATS[k] = 0
        t0 = time.perf_counter()
        out = model(inp, labels=tgt, use_cache=False)
        loss = out.loss
        loss.backward()
        grad_norm = 0.0
        for p in params:
            if p.grad is not None:
                grad_norm += float(p.grad.detach().pow(2).sum())
        grad_norm = math.sqrt(grad_norm)
        if args.mode == 'full':
            opt.step()
            for m in model.modules():
                if isinstance(m, (IgpuLinear, IgpuConv1D)):
                    m.refresh()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        times.append(dt)
        losses.append(float(loss.detach()))
        if step == 0:
            parity = abs(losses[0] - base_loss)
            print('  parity CPU %.5f vs iGPU %.5f -> |diff| %.2e' % (base_loss, losses[0], parity),
                  flush=True)
        finite = math.isfinite(losses[-1]) and math.isfinite(grad_norm)
        ok &= finite
        if step % args.log_every == 0 or step == args.steps - 1:
            print('  step %3d  loss %.4f  grad_norm %.2e  %.3f s  (igpu %d calls, %.0f ms)'
                  % (step, losses[-1], grad_norm, dt, STATS['calls'], STATS['gemm_s'] * 1e3),
                  flush=True)
    total_s = time.perf_counter() - t_start
    med = statistics.median(times[1:]) if len(times) > 1 else times[0]
    print()
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        trainable = {k: v.detach().cpu() for k, v in model.state_dict().items()
                     if 'lora_' in k or args.mode == 'full'}
        if args.mode == 'full':
            trainable = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(trainable, os.path.join(args.save, 'trainable.pt'))
        with open(os.path.join(args.save, 'train_info.json'), 'w', encoding='utf-8') as f:
            json.dump({'model': args.model, 'mode': args.mode, 'steps': args.steps,
                       'loss0': losses[0], 'loss1': losses[-1], 'arch': model.config.architectures}, f, indent=1)
        print('saved trainable weights ->', args.save)
    parity_v = abs(losses[0] - base_loss) if math.isfinite(base_loss) else float('nan')
    print('RESULT %s | %s | %s | steps=%d | %.3f s/step | loss %.4f -> %.4f | finite=%s | parity=%.2e'
          % (args.model, args.mode, model.config.architectures[0] if getattr(model.config, 'architectures', None) else '?',
             args.steps, med, losses[0], losses[-1], 'YES' if ok else 'NO', parity_v))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--mode', choices=['lora', 'full'], default='lora')
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--log-every', type=int, default=5)
    ap.add_argument('--data', type=str, default=None, help='text file to train on')
    ap.add_argument('--save', type=str, default=None, help='dir to save the trainable weights')
    args = ap.parse_args()
    print('device:', ctx.devices[0].name)
    return train(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
