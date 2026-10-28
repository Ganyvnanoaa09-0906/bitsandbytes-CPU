# -*- coding: utf-8 -*-
"""tune_ar.py — 直接测几个能立刻改的开关，看谁能把 tok/s 推上去。

不看 profiler（torch 2.4.1 在本机段错误），改用「关掉某个开关测吞吐」的消融法。

测的开关：
  1. mkldnn 开/关
  2. SDPA 后端：默认 / 强制 math
  3. 线程数 5 / 6
  4. bf16 autocast（预期更慢，验证一下）
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

T = 256
B = 16


def build():
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=16,
        d_model=512, n_layer=10, n_head=8, n_kv_head=4,
        ffn_dim=2048, head_mode='flat', cond_mode='none',
        pos_mode='2d', token_order='raster',
    )
    return SmallARImageModelV2(cfg)


def run_once(tag, threads=6, mkldnn=True, sdpa=None, b=B, iters=4, warm=2):
    torch.set_num_threads(threads)
    torch.backends.mkldnn.enabled = mkldnn
    m = build()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x = torch.randint(0, 1024, (b, T))
    y = torch.randint(0, 1024, (b, T))

    def step():
        out = m.forward_logits(x)
        gen = out[1]
        logits = gen[0] if isinstance(gen, tuple) else gen
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    try:
        for _ in range(warm):
            step()
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        dt = (time.perf_counter() - t0) / iters
        print('  %-34s %7.3f s/步   %8.1f tok/s' % (tag, dt, b * T / dt), flush=True)
        return dt
    except Exception as e:
        print('  %-34s 失败: %s' % (tag, str(e)[:60]), flush=True)
        return None
    finally:
        del m, opt, x, y


print('=' * 68, flush=True)
print('AR 模型调优消融（b=%d, T=%d, 30M 模型）' % (B, T), flush=True)
print('=' * 68, flush=True)

base = run_once('基线: 6线程 mkldnn开 默认SDPA')

print()
print('--- mkldnn 开关 ---', flush=True)
run_once('6线程 mkldnn【关】', mkldnn=False)
run_once('5线程 mkldnn开', threads=5)
run_once('5线程 mkldnn关', threads=5, mkldnn=False)

print()
print('--- SDPA 后端 ---', flush=True)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import contextlib

    for name, backend in (('math', SDPBackend.MATH),
                          ('flash', SDPBackend.FLASH_ATTENTION),
                          ('efficient', SDPBackend.EFFICIENT_ATTENTION)):
        try:
            torch.set_num_threads(6)
            torch.backends.mkldnn.enabled = True
            m = build()
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
            x = torch.randint(0, 1024, (B, T))
            y = torch.randint(0, 1024, (B, T))

            def step2():
                out = m.forward_logits(x)
                gen = out[1]
                logits = gen[0] if isinstance(gen, tuple) else gen
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with sdpa_kernel(backend):
                for _ in range(2):
                    step2()
                t0 = time.perf_counter()
                for _ in range(4):
                    step2()
                dt = (time.perf_counter() - t0) / 4
            print('  %-34s %7.3f s/步   %8.1f tok/s' % ('SDPA=' + name, dt, B * T / dt), flush=True)
            del m, opt, x, y
        except Exception as e:
            print('  %-34s 失败: %s' % ('SDPA=' + name, str(e)[:60]), flush=True)
except Exception as e:
    print('  sdpa_kernel 不可用:', str(e)[:70], flush=True)

print()
print('--- autocast bf16（预期更慢，验证） ---', flush=True)
try:
    torch.set_num_threads(6)
    m = build()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x = torch.randint(0, 1024, (B, T))
    y = torch.randint(0, 1024, (B, T))

    def step3():
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = m.forward_logits(x)
            gen = out[1]
            logits = gen[0] if isinstance(gen, tuple) else gen
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(2):
        step3()
    t0 = time.perf_counter()
    for _ in range(4):
        step3()
    dt = (time.perf_counter() - t0) / 4
    print('  %-34s %7.3f s/步   %8.1f tok/s' % ('autocast bf16', dt, B * T / dt), flush=True)
except Exception as e:
    print('  autocast 失败:', str(e)[:70], flush=True)

print()
print('DONE', flush=True)
