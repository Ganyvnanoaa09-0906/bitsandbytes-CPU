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
import ctypes
import glob
import json
import math
import os
import statistics
import sys
import time
import urllib.request

import numpy
import pyopencl
import torch
import torch.nn

try:
    from memwatch import check_or_exit as _memcheck
except Exception:
    def _memcheck(tag=''):
        return None

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

MIRROR = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
DLL = os.environ.get('CLBLAST_PATH') or (
    r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
    if os.name == 'nt' else 'libclblast.so')
ROW, NO, YES = 101, 111, 112

CL_OK = True
CL_ERROR = None
try:
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
except Exception as _e:          # no OpenCL device or no CLBlast: CPU backend still works
    CL_OK = False
    CL_ERROR = str(_e)[:300]
    ctx = queue = bl = mf = QP = None

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
    def forward(ctx, x, w, w_dev, want_dw, layout, n_dim, k_dim):
        orig = x.shape
        k = orig[-1]
        n = n_dim
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
        return torch.from_numpy(dx), dW, None, None, None, None, None


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
        y = IgpuFn.apply(x, self.weight, self.w_dev, self.weight.requires_grad, 1,
                         self.out_features, self.in_features)
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
        y = IgpuFn.apply(x, self.weight, self.w_dev, self.weight.requires_grad, 0,
                         self.out_features, self.in_features)
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
        if not self.base_weight.requires_grad:
            self.base_weight = None      # frozen base: keep only the device copy (halves host RAM)

    def forward(self, x):
        if self.w_dev is None:
            self.refresh()
        y = IgpuFn.apply(x, None, self.w_dev, False, 0, self.out_features, self.in_features)
        if self.use_bias:
            y = y + self.base_bias
        return y + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale


class CpuLoRALinear(nn.Module):
    """Pure-torch LoRA wrapper used when no iGPU is available (same math as the iGPU path)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight.detach()
        self.bias = base.bias.detach() if base.bias is not None else None
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        return y + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale


class CpuLoRAConv1D(nn.Module):
    """GPT2/NeoX style Conv1D wrapper for the CPU backend (weight stored (in,out))."""

    def __init__(self, base, r: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features = base.weight.shape[0]
        self.out_features = base.weight.shape[1]
        self.weight = base.weight.detach()
        self.bias = base.bias.detach() if getattr(base, 'bias', None) is not None else None
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale


# ---------------------------------------------------------------------------
# Optimizers: the CPU training scripts offer AdamW / AdamW8bit / Muon; the iGPU path
# runs the optimizer on the CPU too, so expose exactly the same choices here.
# Muon follows train_ar_v2.py: NS5 on 2-D weights whose names contain neither 'emb' nor
# 'head', AdamW(8bit) on everything else. The optimizer is CPU-side in both backends.
# ---------------------------------------------------------------------------
_MUON_A, _MUON_B, _MUON_C = 3.4445, -4.7750, 2.0315


def newton_schulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz quintic orthogonalisation (same rule as train_ar_v2.py)."""
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = _MUON_B * A + _MUON_C * (A @ A)
        X = _MUON_A * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon on 2-D weights, for the iGPU kit (optimizer itself runs on the CPU)."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_iters=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_iters = group['ns_iters']
            wd = float(group.get('weight_decay', 0.0) or 0.0)
            for p in group['params']:
                if p.grad is None:
                    continue
                buf = self.state[p].get('buf')
                if buf is None:
                    buf = torch.zeros_like(p)
                    self.state[p]['buf'] = buf
                buf.mul_(momentum).add_(p.grad)
                d = p.grad.add(buf, alpha=momentum)
                if d.ndim == 2:
                    d = newton_schulz5(d, ns_iters)
                    d = d * max(1.0, d.size(0) / d.size(1)) ** 0.5
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(d, alpha=-lr)
        return loss


class CombinedOptimizer:
    """Run a list of optimizers as one (Muon for 2-D + AdamW for the rest)."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    def step(self, closure=None):
        loss = None
        for i, opt in enumerate(self.optimizers):
            loss = opt.step(closure if i == len(self.optimizers) - 1 else None)
        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {'optimizers': [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state):
        parts = state.get('optimizers', state) if isinstance(state, dict) else state
        for opt, sd in zip(self.optimizers, parts):
            opt.load_state_dict(sd)


def make_optimizer(args, named_params):
    """Build the optimizer from --optimizer; returns (optimizer, description)."""
    name = (getattr(args, 'optimizer', 'adamw') or 'adamw').lower()
    wd = float(getattr(args, 'weight_decay', 0.0) or 0.0)
    if name == 'muon':
        muon_p, other_p = [], []
        for n, p in named_params:
            low = n.lower()
            if p.ndim == 2 and 'emb' not in low and 'head' not in low:
                muon_p.append(p)
            else:
                other_p.append(p)
        opts = []
        rest = 'no other params'
        if muon_p:
            opts.append(Muon(muon_p, lr=args.muon_lr, momentum=args.momentum,
                             ns_iters=args.ns_iters, weight_decay=wd))
        if other_p:
            try:
                import bitsandbytes
                opts.append(bnb.optim.AdamW8bit(other_p, lr=args.lr, weight_decay=wd))
                rest = 'AdamW8bit'
            except Exception:
                opts.append(torch.optim.AdamW(other_p, lr=args.lr, weight_decay=wd))
                rest = 'AdamW (bitsandbytes missing)'
        return CombinedOptimizer(opts), 'Muon NS%d on %d 2-D weights + %s on %d' % (
            args.ns_iters, len(muon_p), rest, len(other_p))
    if name in ('adamw4bit', '4bit'):
        try:
            import bitsandbytes
            return bnb.optim.AdamW4bit([p for _, p in named_params], lr=args.lr,
                                       weight_decay=wd), 'AdamW4bit (bnb)'
        except Exception as exc:
            print('  WARNING: AdamW4bit unavailable (%s); falling back' % str(exc)[:80])
    if name in ('adamw8bit', '8bit'):
        try:
            import bitsandbytes
            return bnb.optim.AdamW8bit([p for _, p in named_params], lr=args.lr,
                                       weight_decay=wd), 'AdamW8bit (bnb)'
        except Exception as exc:
            print('  WARNING: bitsandbytes AdamW8bit unavailable (%s); using torch AdamW'
                  % str(exc)[:80])
    return torch.optim.AdamW([p for _, p in named_params], lr=args.lr,
                             weight_decay=wd), 'AdamW (torch)'


def patch_model_cpu(model: nn.Module, mode: str, r: int, skip_lm_head: bool) -> int:

    count = 0
    def walk(mod, prefix=''):
        nonlocal count
        for name, child in list(mod.named_children()):
            full = prefix + name
            if skip_lm_head and full in ('lm_head',):
                continue
            try:
                from transformers.pytorch_utils import Conv1D
            except Exception:
                Conv1D = ()
            if isinstance(child, nn.Linear):
                new = child if mode == 'full' else CpuLoRALinear(child, r)
                if mode == 'full':
                    for p in child.parameters():
                        p.requires_grad_(True)
                else:
                    setattr(mod, name, new)
                count += 1
            elif Conv1D and isinstance(child, Conv1D):
                if mode == 'full':
                    for p in child.parameters():
                        p.requires_grad_(True)
                else:
                    setattr(mod, name, CpuLoRAConv1D(child, r))
                count += 1
            else:
                walk(child, full + '.')
    walk(model)
    return count


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


def _hub_root() -> str:
    """Hub cache root: BNB_HUB_CACHE wins; otherwise the historical Windows path, or ~/.cache."""
    env = os.environ.get('BNB_HUB_CACHE')
    if env:
        return env
    if os.name == 'nt':
        return r'D:\deepsleep\hub_models'
    return os.path.join(os.path.expanduser('~'), '.cache', 'bnb_hub_models')


_WEIGHT_FILES = ('model.safetensors', 'model.safetensors.index.json', 'pytorch_model.bin')


def resolve_model(spec: str) -> str:
    """Local path, or download the needed files from the Hub mirror into the cache root."""
    if os.path.isdir(spec):
        return spec
    local = os.path.join(_hub_root(), spec.replace('/', '__'))
    if os.path.isdir(local) and os.path.exists(os.path.join(local, 'config.json')) and any(
            os.path.exists(os.path.join(local, f)) for f in _WEIGHT_FILES):
        return local
    os.makedirs(local, exist_ok=True)
    files = ['config.json', 'generation_config.json', 'model.safetensors',
             'model.safetensors.index.json', 'pytorch_model.bin', 'tokenizer.json',
             'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
             'tokenizer.model']
    failed = []
    for f in files:
        dst = os.path.join(local, f)
        if os.path.exists(dst):
            continue
        url = '%s/%s/resolve/main/%s' % (MIRROR, spec, f)
        ok = False
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp, open(dst, 'wb') as out:
                    out.write(resp.read())
                ok = True
                break
            except Exception:
                if os.path.exists(dst):
                    os.remove(dst)
                time.sleep(2)
        if not ok:
            failed.append(f)
    # Required files must exist: config.json always, and at least one weight shard/file.
    missing = []
    if not os.path.exists(os.path.join(local, 'config.json')):
        missing.append('config.json')
    if not any(os.path.exists(os.path.join(local, f)) for f in _WEIGHT_FILES):
        missing.append('model weights (model.safetensors / pytorch_model.bin)')
    if missing:
        raise RuntimeError(
            'cannot download model %r from %s: missing %s. Tried into %s; failed files: %s. '
            'Check the model id / network, or pass a local directory.'
            % (spec, MIRROR, ', '.join(missing), local, ', '.join(failed) or 'none'))
    return local


def _load_tokenizer(path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        return None


def make_data(args, model):
    """Task-aware batch: causal LM (input/target shifted) or masked LM (15% random masks)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    vocab = getattr(cfg, 'vocab_size', None) or getattr(getattr(cfg, 'text_config', cfg), 'vocab_size', 32000)
    tok = _load_tokenizer(args.model_path) if (args.data or args.task == 'mlm') else None
    if args.data:
        if tok is None:
            raise RuntimeError(
                '--data needs a tokenizer saved in the model directory (%s); none could be '
                'loaded. Point --model at a local checkpoint that includes tokenizer files.'
                % args.model_path)
        text = open(args.data, encoding='utf-8', errors='replace').read()
        toks = tok(text)['input_ids']
        if len(toks) < args.tokens + 2:
            raise SystemExit('data file too short after tokenisation: %d tokens' % len(toks))
        g = np.random.default_rng(1234)
        win = args.tokens + 1
        seqs = [toks[int(g.integers(0, len(toks) - win)):][:win] for _ in range(args.batch)]
        ids = torch.tensor(seqs, dtype=torch.long)
    else:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, min(vocab, 4000), (args.batch, args.tokens + 1), generator=g)
    if args.task == 'mlm':
        input_ids = ids[:, :-1].clone()
        labels = input_ids.clone()
        mask_id = getattr(tok, 'mask_token_id', None) if tok is not None else None
        if mask_id is None:
            mask_id = 103
        rng = np.random.default_rng(1234)
        prob = rng.random(input_ids.shape)
        mask_pos = prob < 0.15
        input_ids[mask_pos] = mask_id
        labels[~mask_pos] = -100
        attn = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attn}
    # Causal-LM labels must align with the inputs: HuggingFace's loss shifts the logits
    # internally. Passing pre-shifted next tokens here shifts twice and trains t+2
    # prediction -- a real bug found while validating the q3bit trainer (2026-10-20).
    return {'input_ids': ids[:, :-1], 'labels': ids[:, :-1]}


def run_loss(model, task, batch):
    if task == 'mlm':
        return model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                     labels=batch['labels']).loss
    return model(batch['input_ids'], labels=batch['labels'], use_cache=False).loss


def train(args) -> int:
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
    torch.set_num_threads(6)
    path = resolve_model(args.model)
    args.model_path = path
    print('model dir:', path)
    task = args.task
    model = None
    if task in ('auto', 'clm'):
        try:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                         local_files_only=True)
            task = 'clm'
        except Exception as e:
            if args.task == 'clm':
                raise
            print('  causal-LM load failed (%s); trying masked-LM' % str(e)[:80])
    if model is None:
        try:
            model = AutoModelForMaskedLM.from_pretrained(path, torch_dtype=torch.float32,
                                                         local_files_only=True)
            task = 'mlm'
        except Exception as e:
            print()
            print('ERROR: cannot load %s as a text language model.' % args.model)
            print('  tried AutoModelForCausalLM and AutoModelForMaskedLM; last error: %s' % str(e)[:250])
            print('  This trainer supports any checkpoint those two can load (causal LM or masked LM).')
            print('  Multimodal checkpoints / custom quantized formats need their own loader;')
            print('  see gt1030/README_IGPU_TRAIN.md for the supported matrix.')
            return 2
    args.task = task
    print('  task:', task)
    model.eval()
    if args.mode == 'full':
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    backend = args.backend
    if backend == 'auto':
        backend = 'igpu' if CL_OK else 'cpu'
    if backend == 'igpu' and not CL_OK:
        print()
        print('ERROR: iGPU backend requested but OpenCL/CLBlast is unavailable:')
        print('  %s' % CL_ERROR)
        print('  Retry with --backend cpu (pure torch) on machines without an OpenCL device,')
        print('  or install an OpenCL ICD + libclblast and set CLBLAST_PATH.')
        return 2
    args.backend = backend
    print('  backend:', backend + (' (OpenCL device %s)' % ctx.devices[0].name if backend == 'igpu' else ' (no OpenCL needed)'))
    skip_lm = bool(getattr(model.config, 'tie_word_embeddings', False))
    if getattr(args, 'grad_checkpoint', False):
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            print('  gradient checkpointing: enabled')
        except Exception as exc:
            print('  WARNING: gradient checkpointing unavailable: %s' % str(exc)[:120])
    # base_loss above needs eval(); training itself must run in train mode (dropout on).
    model.train()
    total = sum(p.numel() for p in model.parameters())      # before patching: base weights leave Parameters
    batch = make_data(args, model)
    # CPU reference on the same batch, before patching: the iGPU first-step loss must match it
    with torch.no_grad():
        try:
            base_loss = float(run_loss(model, task, batch))
        except Exception:
            base_loss = float('nan')
    if backend == 'igpu':
        n = patch_model(model, args.mode, args.lora_r, skip_lm)
    else:
        n = patch_model_cpu(model, args.mode, args.lora_r, skip_lm)
    if backend == 'igpu' and args.mode == 'lora':
        # upload every frozen base weight once and drop the CPU copies (saves ~model-size host RAM)
        for m in model.modules():
            if isinstance(m, IgpuLoRALinear) and m.w_dev is None:
                m.refresh()
        import gc
        _gc.collect()
        print('  froze base on device; freed CPU copies of base weights')
    _uniq = {id(p): p for p in model.parameters() if p.requires_grad}
    trainable = sum(p.numel() for p in _uniq.values())
    if args.resume:
        try:
            state = torch.load(os.path.join(args.resume, 'trainable.pt'),
                               map_location='cpu', weights_only=True)
        except TypeError:  # very old torch without weights_only
            state = torch.load(os.path.join(args.resume, 'trainable.pt'), map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        print('  resumed %d tensors from %s (missing %d, unexpected %d)'
              % (len(state), args.resume, len(missing), len(unexpected)))
    print('architecture: %s | params %.2f M | trainable %.2f M (%.1f%%) | iGPU linears %d%s'
          % (model.config.architectures[0] if getattr(model.config, 'architectures', None) else '?',
             total / 1e6, trainable / 1e6, 100.0 * trainable / max(1, total), n,
             ' (lm_head kept on CPU: tied embeddings)' if skip_lm else ''))
    # De-duplicate: tied embeddings (and a tied lm_head) expose the same Parameter twice in
    # model.parameters(); passing it twice makes the optimizer apply two updates per step.
    seen_ids, named = set(), []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen_ids:
            seen_ids.add(id(p))
            named.append((n, p))
    params = [p for _, p in named]
    opt, opt_desc = make_optimizer(args, named)
    print('  optimizer:', opt_desc, flush=True)
    if args.resume:
        opt_path = os.path.join(args.resume, 'optimizer.pt')
        if os.path.exists(opt_path):
            try:
                try:
                    state = torch.load(opt_path, map_location='cpu', weights_only=True)
                except TypeError:
                    state = torch.load(opt_path, map_location='cpu')
                opt.load_state_dict(state)
                print('  resumed optimizer state from', opt_path)
            except Exception as exc:
                print('  WARNING: could not load optimizer state (%s); weights-only resume'
                      % str(exc)[:100])
    losses, times = [], []
    ok = True
    t_start = time.perf_counter()
    for step in range(args.steps):
        for k in STATS:
            STATS[k] = 0
        t0 = time.perf_counter()
        out = run_loss(model, task, batch)
        loss = out
        loss.backward()
        grad_norm = 0.0
        for p in params:
            if p.grad is not None:
                grad_norm += float(p.grad.detach().pow(2).sum())
        grad_norm = math.sqrt(grad_norm)
        opt.step()
        if backend == 'igpu' and args.mode == 'full':
            for m in model.modules():
                if isinstance(m, (IgpuLinear, IgpuConv1D)):
                    m.refresh()
        opt.zero_grad(set_to_none=True)
        _memcheck('step')
        dt = time.perf_counter() - t0
        times.append(dt)
        losses.append(float(loss.detach()))
        if step == 0:
            parity = abs(losses[0] - base_loss)
            if args.resume:
                print('  parity skipped: resumed weights, base-model reference does not apply',
                      flush=True)
            else:
                print('  parity CPU %.5f vs iGPU %.5f -> |diff| %.2e'
                      % (base_loss, losses[0], parity), flush=True)
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
        torch.save(opt.state_dict(), os.path.join(args.save, 'optimizer.pt'))
        with open(os.path.join(args.save, 'train_info.json'), 'w', encoding='utf-8') as f:
            json.dump({'model': args.model, 'mode': args.mode, 'steps': args.steps,
                       'optimizer': args.optimizer, 'lr': args.lr,
                       'loss0': losses[0], 'loss1': losses[-1], 'arch': model.config.architectures}, f, indent=1)
        print('saved trainable weights ->', args.save)
    resume_mode = bool(args.resume)
    parity_v = abs(losses[0] - base_loss) if (math.isfinite(base_loss) and not resume_mode) else float('nan')
    print('RESULT %s | %s | %s | optimizer=%s | steps=%d | %.3f s/step | loss %.4f -> %.4f | finite=%s | parity=%s'
          % (args.model, args.mode, model.config.architectures[0] if getattr(model.config, 'architectures', None) else '?',
             args.optimizer, args.steps, med, losses[0], losses[-1], 'YES' if ok else 'NO',
             'n/a (resume)' if resume_mode else '%.2e' % parity_v))
    if args.check:
        good = ok and (resume_mode or (math.isfinite(parity_v) and parity_v < 1e-2))
        print('CHECK:', 'PASS' if good else 'FAIL')
        return 0 if good else 1
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--mode', choices=['lora', 'full'], default='lora')
    ap.add_argument('--task', choices=['auto', 'clm', 'mlm'], default='auto')
    ap.add_argument('--backend', choices=['auto', 'igpu', 'cpu'], default='auto')
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--optimizer', choices=['adamw', 'adamw8bit', 'adamw4bit', 'muon'], default='adamw',
                    help='same choices as the CPU training scripts; runs on the CPU in both backends')
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--momentum', type=float, default=0.95, help='Muon momentum')
    ap.add_argument('--muon-lr', type=float, default=0.02, help='Muon learning rate (2-D weights)')
    ap.add_argument('--ns-iters', type=int, default=5, help='Muon Newton-Schulz iterations')
    ap.add_argument('--log-every', type=int, default=5)
    ap.add_argument('--data', type=str, default=None, help='text file to train on')
    ap.add_argument('--save', type=str, default=None, help='dir to save the trainable weights')
    ap.add_argument('--resume', type=str, default=None, help='dir with trainable.pt to resume from')
    ap.add_argument('--check', action='store_true', help='verify CPU/iGPU parity and exit')
    ap.add_argument('--grad-checkpoint', action='store_true', help='enable HF gradient checkpointing')
    args = ap.parse_args()
    _memcheck('start')
    if args.check:
        args.steps = min(args.steps, 2)
        args.log_every = 1
    if CL_OK:
        print('device:', ctx.devices[0].name)
    else:
        print('device: none (OpenCL unavailable -> CPU backend)')
        if args.backend == 'igpu':
            print('  (requested igpu but no OpenCL device; will fail with instructions)')
    rc = train(args)
    return rc


if __name__ == '__main__':
    sys.exit(main() or 0)
