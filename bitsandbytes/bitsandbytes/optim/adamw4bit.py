"""AdamW4bit -- a self-contained 4-bit AdamW for CPU, calling the fork's kernel.

Why not wire optim_bits=4 through bitsandbytes' own dispatch: that path needs a
new branch in optim/optimizer.py, a new op in _ops.py, a new kernel in
backends/cpu/ops.py and a new registration -- four layers of plumbing for the
same DLL call. Since `lib` is a ctypes handle, calling
coptimizer_update_4bit_blockwise_cpu directly is equivalent and lands in one
file, which is what makes it usable from a training script today.

What it buys (measured, see DESIGN_4BIT_OPTIMIZER.md):
    fp32 AdamW state   8.000 bytes/param
    4-bit state        1.031 bytes/param    -> 7.8x less
    100M model         800 MB -> 103 MB

What it does NOT buy: speed. The kernel currently has no AVX2 path, so it runs
about 4x slower than the 8-bit AVX2 kernel. The optimizer is 1-4% of a training
step, so that is a few percent of wall clock -- acceptable for trying it out, and
the AVX2 version is the next piece of work.

Storage contract (must match cpu_ops.cpp):
  state1/state2 : ceil(numel/2) uint8 each, two 4-bit codes per byte,
                  HIGH nibble = even index (verified against avx2_gemv_4bit.h)
  absmax1/absmax2: one fp32 per 256 elements (kOptBlockSize = 256, cpu_ops.cpp:2458)
  qmap          : 256 fp32; only the first 16 entries are read
"""
from __future__ import annotations

import ctypes
import os

import torch

def _find_dll():
    """Locate a DLL that ACTUALLY EXPORTS the 4-bit optimizer, and load it.

    Returns (path, lib). The symbol check is the point: 'the file exists' is not
    enough, because more than one libbitsandbytes_cpu.dll can be reachable -- a
    source checkout, a stale site-packages install, a leftover build. Asking
    bitsandbytes for its own directory is the best first guess, but in a source
    checkout `import bitsandbytes` can resolve to an OLD site-packages install
    that predates this kernel, and loading that fails at the first call.

    Two earlier versions of this function picked a path without checking the
    symbol; both were wrong and both were masked by a fallback. Now a candidate
    is only accepted after the symbol resolves.
    """
    here = os.path.dirname(os.path.abspath(__file__))

    def _try(path):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return None
        try:
            lib = ctypes.CDLL(path)
            lib.coptimizer_update_4bit_blockwise_cpu      # ★ 唯一可信的判据
            return path, lib
        except Exception:
            return None

    # ★ 顺序与"何时 import bitsandbytes"都是关键，原因不直观：
    #   Windows 的 LoadLibrary 按【模块基名】去重 —— 一旦 libbitsandbytes_cpu.dll
    #   这个基名被某个进程内已加载的模块占用（例如先 import bitsandbytes 拉起的
    #   一个旧 site-packages 安装），之后所有同名 CDLL 调用都会拿回那个旧模块，
    #   即使路径完全不同。表现就是"文件明明有这个符号，加载后却没有"。
    #   ⇒ 所以：先按路径试源码树自己的布局，且【在此期间绝不 import bitsandbytes】；
    #     只有全部失败后，才去问已导入的包要它自己的目录。
    for rel in (os.path.join('bitsandbytes', 'bitsandbytes'),
                os.path.join('bitsandbytes'),
                ''):
        got = _try(os.path.join(here, rel, 'libbitsandbytes_cpu.dll'))
        if got:
            return got

    try:
        import bitsandbytes as _b
        got = _try(os.path.join(os.path.dirname(os.path.abspath(_b.__file__)),
                                'libbitsandbytes_cpu.dll'))
        if got:
            return got
    except Exception:
        pass
    return '', None


_dll_path, _lib = _find_dll()
if _lib is None:
    raise ImportError(
        'no libbitsandbytes_cpu.dll exporting coptimizer_update_4bit_blockwise_cpu was '
        'found. Tried:\n  ' + '\n  '.join(_find_dll.__doc__.splitlines()[:1]) +
        '\nBuild it with build_manual/build_manual.bat (Windows) and retry.')

_F = ctypes.c_float
_U8P = ctypes.POINTER(ctypes.c_ubyte)
_FP = ctypes.POINTER(_F)

_update = _lib.coptimizer_update_4bit_blockwise_cpu
_update.restype = None
_update.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, _U8P, _U8P,
    _F, _F, _F, _F, _F, ctypes.c_int, _F, _FP, _FP, _FP, _FP,
    _F, _F, ctypes.c_int, ctypes.c_longlong, ctypes.c_int,
]

BLOCKSIZE = 256          # must equal kOptBlockSize (cpu_ops.cpp:2458)
OPT_ADAM = 0


def _make_qmap() -> torch.Tensor:
    """16 symmetric levels, padded to 256 entries."""
    q = torch.empty(256, dtype=torch.float32)
    for i in range(256):
        q[i] = (i & 0x0F) * (2.0 / 15.0) - 1.0
    return q


class AdamW4bit(torch.optim.Optimizer):
    """CPU-only 4-bit blockwise AdamW. Drop-in for torch.optim.AdamW on fp32 params."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, blocksize=BLOCKSIZE):
        if blocksize != BLOCKSIZE:
            raise ValueError('blocksize must be %d (compiled into cpu_ops.cpp)'
                             % BLOCKSIZE)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        self._qmap = _make_qmap()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.dtype != torch.float32:
                    raise TypeError('AdamW4bit supports fp32 params only '
                                    '(kernel dtype 0); got %s' % p.dtype)
                g = p.grad
                if not g.is_contiguous():
                    g = g.contiguous()
                n = p.numel()
                nb = (n + BLOCKSIZE - 1) // BLOCKSIZE

                st = self.state[p]
                if 'state1' not in st:
                    st['state1'] = torch.zeros((n + 1) // 2, dtype=torch.uint8)
                    st['state2'] = torch.zeros((n + 1) // 2, dtype=torch.uint8)
                    st['absmax1'] = torch.zeros(nb, dtype=torch.float32)
                    st['absmax2'] = torch.zeros(nb, dtype=torch.float32)
                    st['step'] = 0
                st['step'] += 1

                _update(
                    OPT_ADAM,
                    g.data_ptr(), p.data_ptr(),
                    ctypes.cast(st['state1'].data_ptr(), _U8P),
                    ctypes.cast(st['state2'].data_ptr(), _U8P),
                    _F(b1), _F(b2), _F(0.0), _F(0.0), _F(eps),
                    st['step'], _F(lr),
                    ctypes.cast(self._qmap.data_ptr(), _FP),
                    ctypes.cast(self._qmap.data_ptr(), _FP),
                    ctypes.cast(st['absmax1'].data_ptr(), _FP),
                    ctypes.cast(st['absmax2'].data_ptr(), _FP),
                    _F(wd), _F(1.0), 0, n, 0,
                )
        return loss

    def state_bytes(self) -> int:
        """Actual optimizer-state bytes held (the metric this exists for)."""
        tot = 0
        for st in self.state.values():
            for k in ('state1', 'state2', 'absmax1', 'absmax2'):
                if k in st:
                    tot += st[k].numel() * st[k].element_size()
        return tot
