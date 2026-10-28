# -*- coding: utf-8 -*-
"""自动选定最优线程数 —— 同时满足"可移植"要求。

背景（本机实测）：5 线程比 6 线程快 5.9%（15W 功耗墙）。但训练脚本默认写死 6。
写死任何具体数字都不对：换一台 CPU 最优值就变了。所以这里【实测选优】，
结果缓存到文件，只在首次运行花几秒。

为什么要缓存：每次训练启动都跑一遍基准测试会浪费时间。
缓存 key 里带上 CPU 标识和核数，换机器自动失效。
"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import json
import os
import platform
import time

import torch

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.autotune_threads.json')


def _cpu_key():
    return "%s|%s|%s" % (platform.machine(), os.cpu_count(),
                         (torch.get_num_threads(), torch.__version__.split('+')[0]))


def _bench(nthreads, n=768, duration=4.0, verbose=False):
    """【稳态】吞吐，不是爆发吞吐。

    为什么要这样（本机踩过的坑）：短基准测的是爆发峰值，选出来的是 6 线程（320 GF）；
    但真实训练是持续负载，会撞 15W 功耗墙，此时 5 线程反而比 6 快约 6%。
    所以必须跑到稳态：先热身，再在【后半段】取平均，前半段丢弃。
    """
    torch.set_num_threads(nthreads)
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    torch.mm(a, b)

    t0 = time.perf_counter()
    half = t0 + duration / 2
    n_first = 0
    while time.perf_counter() < half:          # 前 50%：热身 + 让功耗墙生效
        torch.mm(a, b)
        n_first += 1

    t1 = time.perf_counter()
    n_second = 0
    while time.perf_counter() - t1 < duration / 2:
        torch.mm(a, b)
        n_second += 1
    dt = time.perf_counter() - t1

    gf = n_second * 2.0 * n ** 3 / dt / 1e9
    if verbose:
        print("    %d 线程: 前半 %d 次 / 后半 %d 次  -> %.1f GFLOP/s"
              % (nthreads, n_first, n_second, gf))
    return gf


def best_threads(verbose=True, force=False, duration=4.0):
    """返回【稳态】实测最优线程数。带缓存。"""
    key = _cpu_key() + "|steady%.1f" % duration
    if not force:
        try:
            with open(CACHE, encoding='utf-8') as f:
                d = json.load(f)
            if d.get('key') == key:
                if verbose:
                    print('[tune] 使用缓存的线程数 = %d (%.1f GFLOP/s 稳态)'
                          % (d['threads'], d['gflops']), flush=True)
                return d['threads']
        except Exception:
            pass

    ncpu = os.cpu_count() or 4
    cands = sorted({max(1, min(ncpu, n)) for n in (1, 2, 3, 4, 5, 6, 8, ncpu)})
    res = {}
    for n in cands:
        try:
            res[n] = _bench(n, duration=duration)
        except Exception:
            pass
    if not res:
        return max(1, (os.cpu_count() or 4) // 2)
    best = max(res, key=res.get)
    if verbose:
        print('[tune] 稳态线程数实测（越大越好）: %s'
              % "  ".join("%d:%.1f" % (k, v) for k, v in sorted(res.items())), flush=True)
        print('[tune] => 选 %d 线程 (%.1f GFLOP/s 稳态)' % (best, res[best]), flush=True)
    try:
        with open(CACHE, 'w', encoding='utf-8') as f:
            json.dump({'key': key, 'threads': best, 'gflops': res[best], 'all': res}, f)
    except Exception:
        pass
    return best


if __name__ == '__main__':
    print("CPU: %s  cores=%s" % (platform.processor(), os.cpu_count()))
    t = best_threads(force=True)
    print("最优线程数 = %d" % t)
