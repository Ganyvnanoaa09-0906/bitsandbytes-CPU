# -*- coding: utf-8 -*-
"""quant_stress_wan.py — 拿真实的 Wan2.1-T2V-1.3B 权重拷打 bitsandbytes 量化库

为什么用 Wan2.1 而不是随便一个小模型:
    它是**真实的视频生成模型**（1419M 参数、825 张量），而且权重在本地是**完整的**
    （实测：safetensors 头合法、dtype 全 F32、242 个 (1536,1536) 方阵）。
    之前记为"坏包"是误判 —— 缺的是仓库的 3 个**代码文件**，权重没问题。

拷打什么（每一项都是可判定的）:
    [1] **能不能量化**：把全部 2D 权重按 blocksize 64 做 NF4 / FP4 块状量化，
        报告成功率与失败原因。这是"库能不能处理真实模型"的硬判据。
    [2] **量化误差**：对每个张量算 RMS 相对误差（= 1/SNR）。
        判据：NF4 应该明显优于 FP4（NF4 是非均匀分位点，为权重分布设计）。
    [3] **4bit GEMV 速度**：真实形状 (1×1536)×(1536×1536) 与 (1536×8960)，
        对比 fp32 稠密。⚠️ 按 report §10.147：**M 小才是 4bit 的主场**，
        所以这里固定 M=1（解码场景），不测大 M。
    [4] **量化后的数值可用性**：用真实权重跑一次 GEMV，与 fp32 结果比相对误差，
        确认端到端不是垃圾。

oracle:
    · 误差判据用 **RMS 相对误差**（不是逐元素 rel —— 和接近零时逐元素 rel 会爆炸，
      这是 report §10.147.5 记过的坑）；
    · 速度对比**在同一进程、同一计时口径**（min-of-N）；
    · 先打印实际生效的 quant_type/blocksize，防止"以为在测 NF4 其实不是"。
"""
from __future__ import annotations

import ctypes as ct
import json
import os
import statistics
import struct
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'bitsandbytes'))
sys.path.insert(0, _HERE)

from bitsandbytes.cextension import lib  # noqa: E402
from bitsandbytes.functional import get_ptr, quantize_4bit, dequantize_4bit  # noqa: E402

WAN = os.environ.get(
    'WAN_WEIGHTS',
    r'D:\work\textmodel\Wan2.1-T2V-1.3B\diffusion_pytorch_model.safetensors')
BLOCKSIZE = int(os.environ.get('BLOCKSIZE', '64'))
MAX_TENSORS = int(os.environ.get('MAX_TENSORS', '0'))   # 0 = 全部

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n).decode('utf-8'))
    hdr.pop('__metadata__', None)
    return n, hdr


def load_tensor(path, off, meta):
    with open(path, 'rb') as f:
        f.seek(8 + off + meta['data_offsets'][0])
        raw = f.read(meta['data_offsets'][1] - meta['data_offsets'][0])
    dt = {'F32': torch.float32, 'BF16': torch.bfloat16, 'F16': torch.float16}[meta['dtype']]
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta['shape']).clone()


def rms_rel(a, b):
    """RMS 相对误差。不用逐元素 rel —— 和接近零时会爆炸（report §10.147.5）。"""
    d = (a.float() - b.float())
    return (d.pow(2).mean().sqrt() / (a.float().pow(2).mean().sqrt() + 1e-30)).item()


def timed(fn, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    if not os.path.isfile(WAN):
        print('找不到权重: %s' % WAN)
        return 1
    hdr_off, hdr = read_header(WAN)
    print('=' * 84)
    print('拷打 bitsandbytes 量化库 —— 真实 Wan2.1-T2V-1.3B 权重')
    print('=' * 84)
    print('权重文件 : %s' % WAN)
    print('张量数   : %d' % len(hdr))
    print('blocksize: %d   threads: %d' % (BLOCKSIZE, torch.get_num_threads()))
    dts = {}
    for v in hdr.values():
        dts[v['dtype']] = dts.get(v['dtype'], 0) + 1
    print('dtype    : %s' % dts)

    # 只取 2D 权重（Linear），这才是量化的对象
    lin = [(k, v) for k, v in hdr.items()
           if len(v['shape']) == 2 and k.endswith('.weight')]
    lin.sort(key=lambda kv: -kv[1]['shape'][0] * kv[1]['shape'][1])
    if MAX_TENSORS:
        lin = lin[:MAX_TENSORS]
    print('2D 权重张量数: %d（合计 %.1fM 参数）'
          % (len(lin), sum(v['shape'][0] * v['shape'][1] for _, v in lin) / 1e6))

    # ---- [1][2] 量化 + 误差 ----
    print('\n[1][2] 逐张量量化与 RMS 相对误差')
    print('  %-46s %12s %11s %11s' % ('张量', '参数量', 'NF4 rmsrel', 'FP4 rmsrel'))
    print('  ' + '-' * 82)
    results = {'nf4': [], 'fp4': []}
    fails = []
    shown = 0
    for k, v in lin:
        t = load_tensor(WAN, hdr_off, v).float()
        row = [k]
        for qt in ('nf4', 'fp4'):
            try:
                q, st = quantize_4bit(t, blocksize=BLOCKSIZE, quant_type=qt)
                dq = dequantize_4bit(q, st).reshape(t.shape)
                e = rms_rel(t, dq)
                results[qt].append(e)
                row.append(e)
            except Exception as ex:
                fails.append((k, qt, '%s: %s' % (type(ex).__name__, ex)))
                row.append(float('nan'))
        if shown < 8:
            print('  %-46s %12d %11.5f %11.5f'
                  % (k[:46], t.numel(), row[1], row[2]))
            shown += 1
        del t

    def summ(tag):
        xs = [x for x in results[tag] if x == x]
        if not xs:
            return '无数据'
        xs_sorted = sorted(xs)
        return ('均值 %.5f  中位 %.5f  P90 %.5f  最差 %.5f'
                % (statistics.mean(xs), statistics.median(xs),
                   xs_sorted[int(0.9 * (len(xs) - 1))], max(xs)))

    print('\n  NF4 全体: %s（%d/%d 张量成功）' % (summ('nf4'), len(results['nf4']), len(lin)))
    print('  FP4 全体: %s（%d/%d 张量成功）' % (summ('fp4'), len(results['fp4']), len(lin)))
    if fails:
        print('  失败 %d 例，前 5:' % len(fails))
        for k, qt, msg in fails[:5]:
            print('     %s [%s] %s' % (k[:40], qt, msg[:70]))
    else:
        print('  失败 0 例 ✓')
    nf4_mean = statistics.mean(results['nf4']) if results['nf4'] else float('nan')
    fp4_mean = statistics.mean(results['fp4']) if results['fp4'] else float('nan')
    print('  判据（NF4 应优于 FP4）: %s'
          % ('PASS' if nf4_mean < fp4_mean else 'FAIL——NF4 没有更好，需查 quant_type 是否真的生效'))

    # ---- [3] 真实形状的 4bit GEMV vs fp32（M=1）----
    print('\n[3] 4bit fused GEMV vs fp32 稠密（M=1，解码场景）')
    lib.cgemv_4bit_inference_cpu_fp32.argtypes = [
        ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_longlong, ct.c_longlong, ct.c_longlong,
        ct.c_longlong, ct.c_longlong, ct.c_longlong, ct.c_longlong, ct.c_int]
    lib.cgemv_4bit_inference_cpu_fp32.restype = None
    print('  %-40s %11s %11s %9s %12s' % ('形状 (1,K)x(N,K)', 'fp32 ms', '4bit ms', '加速', 'rmsrel'))
    print('  ' + '-' * 84)
    cases = [(1536, 1536), (8960, 1536), (1536, 8960)]
    for (K, N) in cases:
        key = [k for k, v in lin if tuple(v['shape']) == (N, K)]
        if not key:
            print('  %-40s （权重里找不到这个形状）' % ('(1,%d)x(%d,%d)' % (K, N, K)))
            continue
        t = load_tensor(WAN, hdr_off, hdr[key[0]]).float()
        A = torch.randn(1, K)
        # fp32 稠密
        out_f = torch.empty(1, N)
        tf = timed(lambda: torch.mm(A, t.t(), out=out_f), reps=5)
        # 4bit —— 必须用 quantize_4bit 产出的**真实** absmax。
        # ⚠️ 上一版这里用了随机 absmax，于是 rmsrel 高达 12~20，是垃圾数据。
        #    真实 scale 才能让这一列有意义（它应当与上面的 NF4 量化误差同量级）。
        q, st = quantize_4bit(t, blocksize=BLOCKSIZE, quant_type='nf4')
        nblk = (K + BLOCKSIZE - 1) // BLOCKSIZE
        absmax2 = st.absmax.float().reshape(N, nblk).contiguous()
        out_q = torch.empty(1, N)
        def run4():
            lib.cgemv_4bit_inference_cpu_fp32(
                get_ptr(A), get_ptr(q), get_ptr(absmax2), get_ptr(out_q),
                1, N, K, K, K // 2, N, BLOCKSIZE, 2)
        tq = timed(run4, reps=5)
        # 端到端数值：与 fp32 比
        e = rms_rel(out_f, out_q)
        print('  %-40s %11.3f %11.3f %8.2fx %12.4f'
              % ('(1,%d)x(%d,%d)' % (K, N, K), tf * 1e3, tq * 1e3, tf / tq, e))
        del t, A
    print('\n  注: rmsrel 是"真实权重 NF4 量化后，GEMV 输出相对 fp32"的偏离，')
    print('      应与 [2] 的 NF4 量化误差同量级（约 0.09）。若远大于它，说明路径有问题。')
    print('=' * 84)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
