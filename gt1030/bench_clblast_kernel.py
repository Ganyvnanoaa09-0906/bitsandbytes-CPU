"""Reuse CLBlast's own OpenCL xgemm kernel source at runtime (pyopencl), tunable defines.

This compiles src/kernels/common.opencl + level3.opencl + xgemm_part1..4 with PRECISION=16
and parameter defines (MWG/NWG/KWG/MDIMC/NDIMC/VWM/VWN/GEMMK), then launches Xgemm directly.
No C++ DLL rebuild.

Goal: find parameters beating CLBlast's own default (~928-957 GFLOPS) toward 1127 on gfx902.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyopencl as cl

ROOT = r'D:\work\clblast_build\CLBlast-master\src\kernels'


def read_src(path):
    s = open(path, encoding='utf-8').read()
    if s.startswith('\ufeff'):
        s = s[1:]
    # raw-string wrappers are lines `R"(` and `)"`; strip markers (safe for these sources)
    s = s.replace('R"(', '').replace(')"', '')
    return s


def build_source():
    common = read_src(ROOT + r'\common.opencl')
    # strip the C++ raw-string wrapper that CLBlast uses for its own includes
    common = common.split('R"(', 1)[-1]
    if common.rstrip().endswith(')"'):
        common = common.rstrip()[:-2]
    parts = [common, read_src(ROOT + r'\level3\level3.opencl')]
    for f in ('xgemm_part1.opencl', 'xgemm_part2.opencl', 'xgemm_part3.opencl', 'xgemm_part4.opencl'):
        parts.append(read_src(ROOT + '\\level3\\' + f))
    return '\n'.join(parts)


SHAPES = {
    'qkv/o 512x4096x4096': (512, 4096, 4096),
    'mlp up 512x12288x4096': (512, 12288, 4096),
    'mlp down 512x4096x12288': (512, 4096, 12288),
    '1.7B qkv 512x2048x2048': (512, 2048, 2048),
    '1.7B up 512x6144x2048': (512, 6144, 2048),
    '1.7B down 512x2048x6144': (512, 2048, 6144),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mwg', type=int, default=64)
    ap.add_argument('--nwg', type=int, default=64)
    ap.add_argument('--kwg', type=int, default=16)
    ap.add_argument('--mdimc', type=int, default=8)
    ap.add_argument('--ndimc', type=int, default=8)
    ap.add_argument('--vwm', type=int, default=1)
    ap.add_argument('--vwn', type=int, default=1)
    ap.add_argument('--gemmk', type=int, default=0)
    ap.add_argument('--iters', type=int, default=5)
    args = ap.parse_args()

    defines = dict(
        PRECISION=16, GEMMK=args.gemmk, MWG=args.mwg, NWG=args.nwg, KWG=args.kwg,
        MDIMC=args.mdimc, NDIMC=args.ndimc, VWM=args.vwm, VWN=args.vwn,
        MDIMA=args.mdimc, NDIMB=args.ndimc,
        ROUTINE_GEMM=1,
    )
    opts = ['-cl-fast-relaxed-math', '-cl-mad-enable'] + ['-D%s=%d' % (k, v) for k, v in defines.items()]
    src = build_source()
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prog = cl.Program(ctx, src).build(options=opts)
    kern = cl.Kernel(prog, 'Xgemm')
    dev = ctx.devices[0]
    print('device %s CU=%d params %s' % (dev.name, dev.max_compute_units,
          ' '.join('%s=%d' % (k, v) for k, v in defines.items())))

    for name, (M, N, K) in SHAPES.items():
        if K % args.kwg != 0:
            print(name, 'skip K not multiple of KWG', K); continue
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((M, K)).astype(np.float32) * 0.02).astype(np.float16)
        b = (rng.standard_normal((K, N)).astype(np.float32) * 0.02).astype(np.float16)
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 2)
        out = np.empty((M, N), dtype=np.float16)
        gws = (int(np.ceil(M / args.mwg)) * args.mdimc, int(np.ceil(N / args.nwg)) * args.ndimc)
        lws = (args.mdimc, args.ndimc)

        def call():
            kern(q, gws, lws, np.int32(M), np.int32(N), np.int32(K),
                 np.float32(1.0), np.float32(0.0),
                 a_buf, b_buf, c_buf, np.int32(0), np.int32(0))

        call(); q.finish()
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter(); call(); q.finish(); samples.append(time.perf_counter() - t0)
        sec = np.median(samples)
        cl.enqueue_copy(q, out, c_buf).wait()
        ref = a.astype(np.float64) @ b.astype(np.float64)
        err = float(np.abs(out.astype(np.float64) - ref).max())
        gflops = 2.0 * M * N * K / sec / 1e9
        print('%-26s %8.1f ms  %8.1f GFLOPS  max|err| %.4e  %.2fx vs1127'
              % (name, sec * 1e3, gflops, err, gflops / 1127.0), flush=True)


if __name__ == '__main__':
    main()
