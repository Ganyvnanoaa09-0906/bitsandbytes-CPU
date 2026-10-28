# -*- coding: utf-8 -*-
"""enable_fused.py — 一键把融合 elementwise 核接到 AR 模型上

用法（在训练脚本 import 模型之后调用）:
    import enable_fused
    enable_fused.enable()          # 只融合 RMSNorm（+5~8%，最稳）
    enable_fused.enable(block=True) # 再加上残差+LayerScale（收益接近）

实测（交替测量 6 轮，配对比值中位）:
    只 RMSNorm          +5.0%（保守）  ~ +8.0%（中位）
    RMSNorm + SwiGLU + 残差  +5.5%（保守）~ +7.1%（中位）
⇒ 两者接近，默认【只上 RMSNorm】（改动最小、最稳）

⚠️ 必须在创建模型【之前】调用（patch 的是类方法，对已建实例也生效，但为清晰起见前置）。
⚠️ checkpoint 兼容：融合核数值误差 ~1e-7，state_dict 键名不变，可安全互载。
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
import functools

_ENABLED = {'rms': False, 'swiglu': False, 'block': False}
_ORIG = {}


def enable(rmsnorm=True, swiglu=False, block=False, verbose=True):
    """把融合核接到 small_image_model_v2 的算子链上。

    rmsnorm: 融合 `w * x * rsqrt(x.pow(2).mean(-1)+eps)` 的 6 算子链 → 1 个核
    swiglu : 融合 `silu(a) * b`（stride 版，零拷贝吃 chunk 视图）
    block  : 融合 `x + ls * y`（残差 + LayerScale）
    """
    import fused_cpu
    import small_image_model_v2

    if not fused_cpu.available():
        raise RuntimeError('DLL 里没有融合核 —— 需要重新编译 bnb（含 cfused_* 导出）')

    # ---- RMSNorm ----
    if rmsnorm and not _ENABLED['rms']:
        _ORIG['rms'] = M.RMSNorm.forward

        def rms_fwd(self, x):
            return fused_cpu.fused_rmsnorm(x, self.w, None, self.eps)

        M.RMSNorm.forward = rms_fwd
        _ENABLED['rms'] = True

    # ---- SwiGLU ----
    if swiglu and not _ENABLED['swiglu']:
        _ORIG['swiglu'] = M.SwiGLU.forward

        def swiglu_fwd(self, x):
            a, b = self.w12(x).chunk(2, dim=-1)
            return self.w3(fused_cpu.fused_swiglu(a, b))

        M.SwiGLU.forward = swiglu_fwd
        _ENABLED['swiglu'] = True

    # ---- Block 残差 ----
    if block and not _ENABLED['block']:
        _ORIG['block'] = M.BlockV2.forward
        _orig_block = _ORIG['block']

        @functools.wraps(_orig_block)
        def block_fwd(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
            # adaLN / cross 模式走原路径（融合核不覆盖那些调制）
            if self.cm == 'adaln' and pooled is not None:
                return _orig_block(self, x, pooled, ctx, cache, append, keys)
            h = self.ln1(x)
            x = fused_cpu.fused_add_scale(
                x, self.attn(h, cache=cache, append=append), self.ls1)
            x = fused_cpu.fused_add_scale(
                x, self.mlp(self.ln2(x)), self.ls2)
            return x

        M.BlockV2.forward = block_fwd
        _ENABLED['block'] = True

    if verbose:
        on = [k for k, v in _ENABLED.items() if v]
        print('[fused] 已启用: %s' % ', '.join(on), flush=True)
    return True


def disable(verbose=True):
    import small_image_model_v2
    if _ENABLED['rms'] and 'rms' in _ORIG:
        M.RMSNorm.forward = _ORIG['rms']
        _ENABLED['rms'] = False
    if _ENABLED['swiglu'] and 'swiglu' in _ORIG:
        M.SwiGLU.forward = _ORIG['swiglu']
        _ENABLED['swiglu'] = False
    if _ENABLED['block'] and 'block' in _ORIG:
        M.BlockV2.forward = _ORIG['block']
        _ENABLED['block'] = False
    if verbose:
        print('[fused] 已关闭', flush=True)


def status():
    return dict(_ENABLED)


if __name__ == '__main__':
    print('enable_fused 模块。用法:')
    print('    import enable_fused; enable_fused.enable()')
