"""Fused Gated DeltaNet (gated delta rule) recurrent kernels for CPU.

A drop-in CPU replacement for ``fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule``
(targets the Qwen3-Next / Qwen3.5 linear-attention layers). On CPU there is no
triton, so fla falls back to a per-timestep eager loop with a T-deep autograd
graph, which is what makes one backward pass take minutes (728 s observed).
The native kernels in ``csrc/cpu_gdn.cpp`` (AVX2/FMA, OpenMP over heads,
chunk-checkpointed backward) fix that: 4-5x faster fwd/bwd, O(ceil(T/C))
checkpoint memory instead of O(T) saved tensors, bit-identical for any thread
count.

Usage:
    from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule
    o, ht = fused_recurrent_gated_delta_rule(q, k, v, beta, g, scale=s,
                                             initial_state=s0,
                                             output_final_state=True)

    # or patch an installed stack once, then run the model as usual:
    from bitsandbytes.gdn_cpu import patch_fla, patch_transformers
    patch_fla(); patch_transformers()

Layouts follow fla: q/k [B,T,H,K], v [B,T,H,V], beta/g [B,T,H],
initial/final state [B,H,K,V]. Contiguous inputs are read in place (zero
copies, saved for backward without duplication - the big memory win on
long sequences); strided inputs fall back to a per-head-contiguous copy.
Inputs may be any mix of fp32/bf16/fp16 for q/k/v (beta/g are converted to
fp32, as the kernel requires), so strided tensors are handled at the
boundary and never silently misread.

Env knobs:
    GDN_CPU_LIB     path to the shared library (default: search next to this
                    file, then the package dir for libgdn_cpu.so /
                    libbitsandbytes_cpu.so)
    GDN_CPU_CHUNK   chunk length C for the backward checkpoints. Unset (the
                    default) auto-sizes C to minimize peak memory
                    (checkpoints = B*H*ceil(T/C)*K*V*4 bytes); set a positive
                    int to pin it.
    GDN_CPU_DISABLE set to 1 to make the op raise, so callers can verify the
                    fallback path still works
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import torch

__all__ = [
    "fused_recurrent_gated_delta_rule",
    "FusedRecurrentGatedDeltaRuleFunction",
    "patch_fla",
    "patch_transformers",
    "load_native",
]

_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_LIB = None


def _candidate_paths() -> list[Path]:
    env = os.environ.get("GDN_CPU_LIB")
    here = Path(__file__).resolve().parent
    pkg = here.parent
    out = []
    if env:
        out.append(Path(env))
    # Windows 上优先 .dll（.so 是 Linux 产物，Windows 载入会以 0xc000012f 崩溃），
    # 非 Windows 依次 .so/.dylib
    dll_names = [
        "libbitsandbytes_cpu.dll",  # CMake CPU backend output on Windows
        "bitsandbytes_cpu.dll",
        "bitsandbytes.dll",
        "gdn_cpu.dll",
    ]
    so_names = [
        "libgdn_cpu.so",
        "libbitsandbytes_cpu.so",
    ]
    if os.name == "nt":
        order = [here / n for n in dll_names] + [pkg / "libbitsandbytes_cpu.dll"]
    else:
        order = [here / n for n in so_names] + [here / "libgdn_cpu.dylib", pkg / "libbitsandbytes_cpu.so"]
    out += order
    return out


def load_native() -> ctypes.CDLL:
    """Load and memoize the native library holding gdn_fwd_cpu/gdn_bwd_cpu."""
    global _LIB
    if _LIB is not None:
        return _LIB
    tried = []
    for p in _candidate_paths():
        if not p.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(p))
        except OSError as e:  # pragma: no cover - environment dependent
            tried.append(f"{p}: {e}")
            continue
        for fn in ("gdn_fwd_cpu", "gdn_bwd_cpu"):
            getattr(lib, fn)  # AttributeError if the symbol is missing
        _bind_signatures(lib)
        _LIB = lib
        return lib
    raise ImportError(
        "gdn_cpu native library not found. Build it with\n"
        "  g++ -O2 -mavx2 -mfma -fopenmp -ffp-contract=off -fPIC -shared \\\n"
        "      -I bitsandbytes/csrc bitsandbytes/csrc/cpu_gdn.cpp \\\n"
        "      -o bitsandbytes/libgdn_cpu.so\n"
        "or point GDN_CPU_LIB at it. Tried: "
        + ("; ".join(tried) if tried else "no candidates existed")
    )


def _bind_signatures(lib: ctypes.CDLL) -> None:
    fptr = ctypes.c_void_p
    ip = ctypes.c_int
    lib.gdn_fwd_cpu.restype = ip
    lib.gdn_fwd_cpu.argtypes = [fptr] * 9 + [ip] * 8   # 9 ptr + B,H,T,K,V,dtype,C,layout
    lib.gdn_bwd_cpu.restype = ip
    lib.gdn_bwd_cpu.argtypes = [fptr] * 14 + [ip] * 8  # 14 ptr + B,H,T,K,V,dtype,C,layout


def _ptr(t: Optional[torch.Tensor]) -> int:
    return t.data_ptr() if t is not None else 0


def _avail_ram_bytes() -> int:
    """Best-effort available RAM (no new deps). psutil 优先（跨平台），
    Linux: MemAvailable; Windows: GlobalMemoryStatusEx; elsewhere assume 8 GB."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MS(ctypes.Structure):
                # 完整 MEMORYSTATUSEX（缺字段 -> GlobalMemoryStatusEx 返回 FALSE）
                _fields_ = [
                    ("dwLength", ctypes.c_uint), ("dwMemoryLoad", ctypes.c_uint),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ]

            ms = _MS(dwLength=ctypes.sizeof(_MS))
            k32 = ctypes.windll.kernel32
            k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MS)]
            k32.GlobalMemoryStatusEx.restype = ctypes.c_int
            if k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullAvailPhys)
        except Exception:
            pass
    return 8 << 30


def _chunk_len(B: int, H: int, T: int, K: int, V: int) -> int:
    """Checkpoint chunk length C.

    GDN_CPU_CHUNK (if set) pins it explicitly. Otherwise minimize peak
    TRAINING memory. Two terms compete:

      - checkpoints B*H*ceil(T/C)*K*V*4 B: alive from each layer's forward
        until that layer's backward, so during training they coexist across
        ALL layers -> shrink with larger C (persistent term);
      - the per-thread chunk-recompute buffer ((C+2)*K*V*4 B): alive only
        while ONE layer's backward runs (transient term) -> grows with C.

    Speed is flat in C over 32..512 (measured), so the persistent term wins:
    take the largest C (<= 512) whose p-thread recompute buffer fits a budget
    of 25% of available RAM (clamped to [512 MB, 4 GB]); floor 64. For
    T=8192, H=16, K=V=128 this gives C=512 -> 16 MB checkpoints/layer vs
    128 MB at C=64 (a 12-layer model saves ~1.3 GB).
    """
    env = os.environ.get("GDN_CPU_CHUNK")
    if env:  # explicit pin (empty/garbage falls back to auto)
        try:
            c = int(env)
        except ValueError:
            c = 0
        if c > 0:
            return c
    p = max(1, min(os.cpu_count() or 1, B * H))
    row = max(4 * K * V, 4)  # bytes of one K*V state row (guard: no /0)
    budget = min(max(_avail_ram_bytes() // 4, 512 << 20), 4 << 30)
    c = budget // (p * row) - 2
    c = max(64, min(c, 512))
    return min(c, max(T, 1))


class FusedRecurrentGatedDeltaRuleFunction(torch.autograd.Function):
    """Autograd wrapper around the C kernels.

    Inputs are already fla-layout; every tensor is made C-contiguous here so
    the raw-pointer kernel can never read strided memory as dense.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,   # [B,T,H,K] (pre-scaled by caller)
        k: torch.Tensor,   # [B,T,H,K]
        v: torch.Tensor,   # [B,T,H,V]
        beta: torch.Tensor,  # [B,T,H]
        g: Optional[torch.Tensor],  # [B,T,H] log decay per step (fp32-ish)
        initial_state: Optional[torch.Tensor],  # [B,H,K,V]
        need_state_grad: bool,
        grad_mode: Optional[bool] = None,  # torch.is_grad_enabled() at apply()
    ):
        if os.environ.get("GDN_CPU_DISABLE"):
            raise RuntimeError("GDN_CPU_DISABLE is set")
        lib = load_native()

        B, T, H, K = q.shape
        V = v.shape[-1]
        dt = q.dtype
        assert dt in _DTYPE_CODE and k.dtype == dt and v.dtype == dt, \
            f"q/k/v must share an fp32/bf16/fp16 dtype, got {q.dtype}/{k.dtype}/{v.dtype}"
        assert q.device.type == "cpu" and k.is_cpu and v.is_cpu

        # -> kernel layout. layout 1 ([B,T,H,D] batch-contiguous, the layout
        # model tensors already come in) is read in place with ZERO copies;
        # layout 0 is the per-head-contiguous [B,H,T,D] packing used when any
        # input is strided. beta/g -> fp32 [B,H,T] in both cases.
        if q.is_contiguous() and k.is_contiguous() and v.is_contiguous():
            qc, kc, vc, layout = q, k, v, 1
        else:
            qc = q.permute(0, 2, 1, 3).contiguous()
            kc = k.permute(0, 2, 1, 3).contiguous()
            vc = v.permute(0, 2, 1, 3).contiguous()
            layout = 0
        bt = beta.permute(0, 2, 1).float().contiguous()
        gt = g.permute(0, 2, 1).float().contiguous() if g is not None \
            else torch.zeros(B, H, T, dtype=torch.float32)
        s0 = initial_state.float().contiguous() if initial_state is not None else None

        C = _chunk_len(B, H, T, K, V)
        nc = (T + C - 1) // C
        o = torch.empty(B, T, H, V, dtype=dt) if layout == 1 \
            else torch.empty(B, H, T, V, dtype=dt)
        sf = torch.empty(B, H, K, V, dtype=torch.float32)
        # Checkpoints are only consumed by the backward. NOTE: inside a
        # classic autograd.Function forward, torch.is_grad_enabled() is ALWAYS
        # False (autograd does not record here), so the decision must come
        # from the caller's grad mode and ctx.needs_input_grad. Skipping the
        # allocation saves B*H*nc*K*V*4 bytes whenever no graph node is
        # created (inference under no_grad, or grad-enabled validation loops
        # where no input requires grad - autograd builds no node there, so
        # backward can never run).
        will_backward = (grad_mode is not False) and any(ctx.needs_input_grad)
        ck = torch.empty(B, H, nc, K, V, dtype=torch.float32) if will_backward else None

        rc = lib.gdn_fwd_cpu(
            _ptr(qc), _ptr(kc), _ptr(vc), _ptr(bt), _ptr(gt), _ptr(s0),
            _ptr(o), _ptr(sf), _ptr(ck), B, H, T, K, V, _DTYPE_CODE[dt], C, layout,
        )
        if rc != 0:
            raise RuntimeError(f"gdn_fwd_cpu failed with code {rc}")

        # Only materialize the dtype-cast final state when the caller asked
        # for it (fla/transformers convention: state dtype matches the input
        # dtype); otherwise hand back the fp32 buffer - the wrapper discards
        # it, and skipping the copy saves B*H*K*V per layer in training.
        s_out = sf.to(dt) if need_state_grad else sf
        ctx.save_for_backward(qc, kc, vc, bt, gt, ck)
        ctx.gdn_meta = (B, H, T, K, V, _DTYPE_CODE[dt], C, dt,
                        need_state_grad, initial_state is not None,
                        g is not None, layout, beta.dtype,
                        g.dtype if g is not None else None)
        return (o if layout == 1 else o.permute(0, 2, 1, 3)), s_out

    @staticmethod
    def backward(ctx, do_: torch.Tensor, ds: Optional[torch.Tensor]):
        lib = load_native()
        qc, kc, vc, bt, gt, ck = ctx.saved_tensors
        (B, H, T, K, V, dtc, C, dt, need_state_grad, had_s0, had_g, layout,
         bdt, gdt) = ctx.gdn_meta
        if ck is None:
            raise RuntimeError(
                "GDN backward called without checkpoints: the forward ran "
                "under torch.no_grad() (or grad was disabled), so no "
                "checkpoint tensor was saved. Re-run the forward with grad "
                "enabled to train."
            )

        if layout == 1:
            # dO in the same [B,T,H,V] batch-contiguous layout: usually the
            # upstream grad already is that (zero copy), otherwise materialize
            doc = do_ if (do_.is_contiguous() and do_.dtype == dt) \
                else do_.to(dt).contiguous()
            dq = torch.empty(B, T, H, K, dtype=dt)
            dk = torch.empty(B, T, H, K, dtype=dt)
            dv = torch.empty(B, T, H, V, dtype=dt)
        else:
            doc = do_.permute(0, 2, 1, 3).to(dt).contiguous()
            dq = torch.empty(B, H, T, K, dtype=dt)
            dk = torch.empty(B, H, T, K, dtype=dt)
            dv = torch.empty(B, H, T, V, dtype=dt)
        # grad of the (possibly dtype-cast) final state; already [B,H,K,V]
        # kernel layout - NO permute here (only [B,T,H,*] tensors get one)
        dsf = None
        if ds is not None:
            dsf = ds.float().contiguous()

        dbeta = torch.empty(B, H, T, dtype=torch.float32)
        dg = torch.empty(B, H, T, dtype=torch.float32)
        dsi = torch.empty(B, H, K, V, dtype=torch.float32)

        rc = lib.gdn_bwd_cpu(
            _ptr(qc), _ptr(kc), _ptr(vc), _ptr(bt), _ptr(gt), _ptr(doc),
            _ptr(dsf), _ptr(ck), _ptr(dq), _ptr(dk), _ptr(dv),
            _ptr(dbeta), _ptr(dg), _ptr(dsi), B, H, T, K, V, dtc, C, layout,
        )
        if rc != 0:
            raise RuntimeError(f"gdn_bwd_cpu failed with code {rc}")

        if layout == 0:  # -> [B,T,H,*]
            dq = dq.permute(0, 2, 1, 3)
            dk = dk.permute(0, 2, 1, 3)
            dv = dv.permute(0, 2, 1, 3)
        # cast the scalar-path grads to the inputs' dtypes (they may be
        # bf16/fp16 in a real model; the kernel always emits fp32)
        dbeta = dbeta.permute(0, 2, 1).to(bdt)
        # g was a None input: autograd REQUIRES None here, not a tensor
        dgt = dg.permute(0, 2, 1).to(gdt) if had_g else None
        ds0 = dsi if had_s0 else None
        return dq, dk, dv, dbeta, dgt, ds0, None, None  # last None: grad_mode


def fused_recurrent_gated_delta_rule(
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    head_first: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
):
    """fla-compatible fused_recurrent_gated_delta_rule for CPU tensors.

    ``q/k`` [B,T,H,K], ``v`` [B,T,H,V], ``beta``/``g`` [B,T,H] (g = per-step
    log decay), ``initial_state`` [B,H,K,V]. ``scale`` multiplies q before the
    recurrence (fla semantics); q/k l2-normalization, when requested, is done
    with regular torch ops so its backward stays exact. ``head_first=True``
    accepts/returns the legacy [B,H,T,D] fla layout. Returns ``(o, final_state)``
    like fla (final_state is None unless output_final_state=True). ``r=`` is
    accepted as an alias for ``q`` (newer fla renamed the first argument).
    """
    if q is None and "r" in kwargs:
        q = kwargs.pop("r")  # fla >= 0.3 calls the query r
    if q is None or k is None or v is None:
        raise TypeError("fused_recurrent_gated_delta_rule() requires q, k, v tensors")
    if initial_state is None and kwargs.get("s0") is not None:
        initial_state = kwargs.pop("s0")  # some callers use s0/h0
    if kwargs:
        raise TypeError(f"unexpected arguments: {sorted(kwargs)}")
    if cu_seqlens is not None:
        raise NotImplementedError("varlen (cu_seqlens) is not supported on the CPU path")
    if head_first:  # legacy fla layout [B,H,T,D] -> [B,T,H,D]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        beta = None if beta is None else beta.transpose(1, 2)
        g = None if g is None else g.transpose(1, 2)
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, \
        f"expected [B,T,H,D], got q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)}"
    assert q.shape == k.shape and q.shape[:3] == v.shape[:3], \
        f"incompatible shapes q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)}"
    if beta is None:
        beta = torch.ones(*q.shape[:3], dtype=q.dtype, device=q.device)
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"scale must be a float, got {type(scale).__name__} (positional "
                "arg #6 is `scale`; `initial_state` is #7)"
            ) from e
    # l2-normalize FIRST, then apply scale: normalize() is scale-invariant,
    # so the reverse order would silently cancel a non-1 scale (transformers'
    # torch fallbacks do normalize-then-scale).
    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
    if scale is not None and scale != 1.0:
        q = q * scale

    o, s = FusedRecurrentGatedDeltaRuleFunction.apply(
        q, k, v, beta, g, initial_state, bool(output_final_state),
        torch.is_grad_enabled(),
    )
    if head_first:
        o = o.transpose(1, 2)
    return o, (s if output_final_state else None)


# ---------------------------------------------------------------------------
# monkey-patch entries: swap in our CPU kernel, keep the original for CUDA
# ---------------------------------------------------------------------------
def patch_fla() -> bool:
    """Replace fla's fused_recurrent_gated_delta_rule with a CPU-dispatching
    wrapper. Returns True if fla was found and patched."""
    try:
        import fla.ops.gated_delta_rule as fgr
    except Exception:
        return False
    orig = getattr(fgr, "fused_recurrent_gated_delta_rule", None)
    if orig is None or getattr(orig, "_gdn_cpu_patched", False):
        return orig is not None

    def cpu_dispatch(*args, **kw):
        # NOTE: never use `kw.get("q") or kw.get("r")` - bool(Tensor) raises
        # for numel > 1 ("Boolean value of Tensor is ambiguous").
        lead = args[0] if args and isinstance(args[0], torch.Tensor) else \
            kw.get("q", kw.get("r"))
        if isinstance(lead, torch.Tensor) and lead.is_cpu:
            return fused_recurrent_gated_delta_rule(*args, **kw)
        return orig(*args, **kw)

    cpu_dispatch._gdn_cpu_patched = True
    fgr.fused_recurrent_gated_delta_rule = cpu_dispatch
    # some versions re-export at package level
    try:
        import fla.ops as fo
        if getattr(fo, "fused_recurrent_gated_delta_rule", None) is orig:
            fo.fused_recurrent_gated_delta_rule = cpu_dispatch
    except Exception:
        pass
    return True


def patch_transformers() -> bool:
    """Patch transformers' Qwen3-Next/Qwen3.5 GatedDeltaNet to use the CPU kernel.

    Returns True if at least one module was found and patched.

    Covers the symbols a GatedDeltaNet forward can dispatch to:
      - ``fused_recurrent_gated_delta_rule`` (older transformers / some forks);
      - ``torch_recurrent_gated_delta_rule`` / ``torch_chunk_gated_delta_rule``
        (transformers 5.15+): both are module attributes that the model's
        ``forward`` looks up at call time, so replacing them redirects the
        torch fallback paths to the fused kernel. The recurrent kernel is
        mathematically identical to the chunked scan, so substituting it for
        the chunk path is exact (fla guarantees the same for its own pair).
    """
    targets = [
        "transformers.models.qwen3_next.modeling_qwen3_next",
        "transformers.models.qwen3_5.modeling_qwen3_5",
    ]
    import importlib
    patched = False
    for name in targets:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue

        def make_dispatch(orig):
            def cpu_dispatch(*args, **kw):
                # `or` on tensors raises for numel > 1; resolve explicitly
                q = args[0] if args and isinstance(args[0], torch.Tensor) else \
                    kw.get("q", kw.get("r"))
                if not (isinstance(q, torch.Tensor) and q.is_cpu):
                    return orig(*args, **kw)
                # varlen (cu_seqlens) is not implemented on the CPU kernel
                if kw.get("cu_seqlens") is not None:
                    return orig(*args, **kw)
                # forward passes a grab-bag of extra kwargs (attn masks,
                # cache, etc.) - hand the kernel only what it understands
                supported = ("q", "r", "k", "v", "beta", "g", "scale",
                             "initial_state", "s0", "output_final_state",
                             "use_qk_l2norm_in_kernel", "head_first")
                clean = {k: v for k, v in kw.items() if k in supported}
                # transformers' torch fallbacks multiply q by 1/sqrt(K) INSIDE
                # the function (the caller never passes `scale`); our kernel
                # does not scale unless asked, so replicate that convention
                # here, otherwise the patched output would be off by 1/sqrt(K)
                if "scale" not in clean:
                    k_t = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) \
                        else kw.get("k")
                    if k_t is not None:
                        clean["scale"] = k_t.shape[-1] ** -0.5
                return fused_recurrent_gated_delta_rule(*args, **clean)
            cpu_dispatch._gdn_cpu_patched = True
            return cpu_dispatch

        for attr in ("fused_recurrent_gated_delta_rule",
                     "torch_recurrent_gated_delta_rule",
                     "torch_chunk_gated_delta_rule"):
            orig = getattr(mod, attr, None)
            if orig is None or getattr(orig, "_gdn_cpu_patched", False):
                continue
            setattr(mod, attr, make_dispatch(orig))
            patched = True
    return patched


# ---------------------------------------------------------------------------
# self-test: python -m bitsandbytes.gdn_cpu
# ---------------------------------------------------------------------------
def _reference(q, k, v, beta, g, s0, do_, ds_final):
    """Naive double-precision per-step reference with exact autograd grads."""
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    b64 = beta.detach().double().requires_grad_(True)
    g64 = g.detach().double().requires_grad_(True) if g is not None else None
    s64 = s0.detach().double().requires_grad_(True) if s0 is not None \
        else torch.zeros(q.shape[0], q.shape[2], q.shape[-1], v.shape[-1],
                         dtype=torch.float64, requires_grad=True)
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = s64.clone()
    outs = []
    for t in range(T):
        eg = g64[:, t].exp() if g64 is not None else 1.0
        S = S * eg[:, :, None, None]
        w = torch.einsum("bhi,bhij->bhj", k64[:, t], S)
        u = v64[:, t] - w
        up = b64[:, t].unsqueeze(-1) * u
        S = S + torch.einsum("bhi,bhj->bhij", k64[:, t], up)
        outs.append(torch.einsum("bhi,bhij->bhj", q64[:, t], S))
    o = torch.stack(outs, dim=1)
    loss = (o * do_.detach().double()).sum()
    if ds_final is not None:
        loss = loss + (S * ds_final.double()).sum()
    grads = torch.autograd.grad(loss, [q64, k64, v64, b64] + ([g64] if g64 is not None else []) + [s64],
                                allow_unused=True)
    return o.detach(), S.detach(), grads


def _selftest() -> int:
    torch.manual_seed(7)
    dev = torch.device("cpu")
    B, T, H, K, V = 2, 33, 3, 16, 24
    fails = 0

    for dt, tol in ((torch.float32, 2e-3), (torch.bfloat16, 4e-2), (torch.float16, 4e-2)):
        q = torch.randn(B, T, H, K, dtype=dt) * K ** -0.5
        k = torch.randn(B, T, H, K, dtype=dt) * K ** -0.5
        v = torch.randn(B, T, H, V, dtype=dt) * 0.3
        beta = torch.rand(B, T, H, dtype=torch.float32) * 0.8 + 0.2
        g = -torch.rand(B, T, H, dtype=torch.float32) * 1.5 - 0.05
        s0 = torch.randn(B, H, K, V, dtype=dt) * 0.2
        do_ = torch.randn(B, T, H, V, dtype=torch.float32)
        dsf = torch.randn(B, H, K, V, dtype=torch.float32) * 0.5

        o, ht = fused_recurrent_gated_delta_rule(
            q, k, v, beta, g, initial_state=s0, output_final_state=True)
        o_ref, ht_ref, grads_ref = _reference(
            q, k, v, beta, g, s0, do_, dsf)

        err_o = (o.float() - o_ref.float()).abs().max().item()
        err_s = (ht.float() - ht_ref.float()).abs().max().item()
        ok = err_o < tol and err_s < tol
        fails += not ok
        print(f"  {str(dt):15s} fwd: |dO|max {err_o:.2e}  |dS|max {err_s:.2e}  {'ok' if ok else 'FAIL'}")

        # backward vs double-precision autograd on the naive reference
        qa = q.clone().detach().requires_grad_(True)
        ka = k.clone().detach().requires_grad_(True)
        va = v.clone().detach().requires_grad_(True)
        ba = beta.clone().requires_grad_(True)
        ga = g.clone().requires_grad_(True)
        sa = s0.clone().detach().requires_grad_(True)
        o3, ht3 = FusedRecurrentGatedDeltaRuleFunction.apply(
            qa, ka, va, ba, ga, sa, True)
        loss3 = (o3.float() * do_).sum() + (ht3.float() * dsf).sum()
        loss3.backward()

        rq, rk, rv, rb, rg, rs = grads_ref
        rel = lambda a, b: ((a.float() - b.float()).abs().max() /
                            (b.float().abs().max() + 1e-6)).item()
        e_q, e_k, e_v = rel(qa.grad, rq), rel(ka.grad, rk), rel(va.grad, rv)
        e_b, e_g, e_s = rel(ba.grad, rb), rel(ga.grad, rg), rel(sa.grad, rs)
        tolg = 3e-2 if dt != torch.float32 else 3e-3
        okg = max(e_q, e_k, e_v, e_b, e_g, e_s) < tolg
        fails += not okg
        print(f"  {str(dt):15s} bwd rel: dq {e_q:.2e} dk {e_k:.2e} dv {e_v:.2e} "
              f"dbeta {e_b:.2e} dg {e_g:.2e} ds0 {e_s:.2e}  {'ok' if okg else 'FAIL'}")

    print("selftest", "PASSED" if fails == 0 else f"FAILED ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
