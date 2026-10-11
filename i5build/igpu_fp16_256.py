"""顺着 wave 数往下推：fp16 操作数 + 256 线程 ⇒ 撞到 40 wave 硬件上限。

LDS 占用（tile 决定，与线程数无关）：
    fp32  2 × 8 × (64+4) × 4 B = 4352 B  ⇒ 7 工作组  ⇒ 7 / 14 / 28 wave（64/128/256 线程）
    fp16  2 × 8 × (64+8) × 2 B = 2304 B  ⇒ 14 工作组 ⇒ 256 线程时 56 wave，**撞 40 上限**
⇒ 若"wave 数即那个变量"，fp16+256 应从 28 涨到 40 wave（理论上限 +43%）。
   代价是 fp16→fp32 的转换指令（已知 fp16 路径的固定开销）。

对照：fp32 256 线程 = 727~795 GFLOPS（占实测上限 960 的 76~83%）。
"""
import os
import shutil
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_fp16w')

HEAD = r'''
#define TS 64
#define TK 8
#define TM 4
#define TN 4
#define NTH_ 256
'''

# fp32 256 线程（基线）
K32 = HEAD + r'''
__kernel void v32(const int M, const int N, const int K,
                  const __global float* restrict A,
                  const __global float* restrict B,
                  __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int t = ty * 16 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int am = t % 64, akp = t / 64;
    const int bk = t / 32, bnp = t % 32;
    for (int k0 = 0; k0 < K; k0 += TK) {
        float2 va = *(const __global float2*)(A + (by+am)*KC + k0 + akp*2);
        As[akp*2+0][am]=va.x; As[akp*2+1][am]=va.y;
        float2 vb = *(const __global float2*)(B + (k0+bk)*NC + bx + bnp*2);
        Bs[bk][bnp*2+0]=vb.x; Bs[bk][bnp*2+1]=vb.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i=0;i<TM;i++) a[i]=As[k][ty*TM+i];
            for (int j=0;j<TN;j++) b[j]=Bs[k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i=0;i<TM;i++) { int rc=by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) { int cc=bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc]=acc[i][j]; } }
}
'''

# fp16 操作数 + 256 线程（LDS 2304 B ⇒ 14 工作组；每工作组 4 wave ⇒ 56，撞 40 上限）
K16 = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
''' + HEAD + r'''
__kernel void v16(const int M, const int N, const int K,
                  const __global half* restrict A,
                  const __global half* restrict B,
                  __global float* restrict C) {
    __local half As[TK][TS + 8];
    __local half Bs[TK][TS + 8];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int t = ty * 16 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int am = t % 64, akp = t / 64;
    const int bk = t / 32, bnp = t % 32;
    for (int k0 = 0; k0 < K; k0 += TK) {
        // half2：4 字节，A 沿 k 连续、B 沿 n 连续，k0+kp*2 与 np*2 均为偶数 ⇒ 对齐
        half2 va = *(const __global half2*)(A + (by+am)*KC + k0 + akp*2);
        As[akp*2+0][am]=va.s0; As[akp*2+1][am]=va.s1;
        half2 vb = *(const __global half2*)(B + (k0+bk)*NC + bx + bnp*2);
        Bs[bk][bnp*2+0]=vb.s0; Bs[bk][bnp*2+1]=vb.s1;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i=0;i<TM;i++) a[i]=(float)As[k][ty*TM+i];
            for (int j=0;j<TN;j++) b[j]=(float)Bs[k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i=0;i<TM;i++) { int rc=by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) { int cc=bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc]=acc[i][j]; } }
}
'''

PROBE = HEAD + r'''
__kernel void probe(const int M, const int N, const int K,
                    const __global float* restrict A,
                    const __global float* restrict B,
                    __global float* dA, __global float* dB) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int t = ty * 16 + tx;
    const int am = t % 64, akp = t / 64;
    const int bk = t / 32, bnp = t % 32;
    float2 va = *(const __global float2*)(A + (by+am)*KC + akp*2);
    As[akp*2+0][am]=va.x; As[akp*2+1][am]=va.y;
    float2 vb = *(const __global float2*)(B + bk*NC + bx + bnp*2);
    Bs[bk][bnp*2+0]=vb.x; Bs[bk][bnp*2+1]=vb.y;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = t; i < 512; i += NTH_) {
        dA[i] = As[i/TS][i%TS];
        dB[i] = Bs[i/TS][i%TS];
    }
}
'''


def build(ctx, tag, src, K, N):
    d = os.path.join(BASE, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        return cl.Program(ctx, src).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable', f'-DKC={K}', f'-DNC={N}'])
    finally:
        os.chdir(cwd)


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    M = N = K = 512
    a = np.arange(M * K, dtype=np.float32).reshape(M, K) % 97 + 0.5
    b = np.arange(K * N, dtype=np.float32).reshape(K, N) % 89 + 0.25
    ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
          cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))

    print('=== 1) 加载探针（fp32-256 的映射，fp16 版共用同一套索引）===')
    dA = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    dB = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    pp = build(ctx, 'probe', PROBE, K, N)
    cl.Kernel(pp, 'probe')(q, (16, 16), (16, 16), np.int32(M), np.int32(N),
                           np.int32(K), ab[0], ab[1], dA, dB)
    q.finish()
    oA = np.empty(512, np.float32); oB = np.empty(512, np.float32)
    cl.enqueue_copy(q, oA, dA).wait(); cl.enqueue_copy(q, oB, dB).wait()
    expA = np.stack([a[0:64, k] for k in range(8)])
    expB = np.stack([b[k, 0:64] for k in range(8)])
    bad = int((oA.reshape(8, 64) != expA).sum() + (oB.reshape(8, 64) != expB).sum())
    print(f'   错格 {bad}  {"✅" if bad == 0 else "❌"}')
    if bad:
        return 1

    print('\n=== 2) 正确性 + 计时 ===')
    print(f"{'形状':>20} {'fp32/28wave':>12} {'fp16/40wave':>12} {'比值':>7}")
    for M2, N2, K2 in ((512, 512, 512), (1024, 1024, 1024), (2048, 512, 512),
                       (2048, 2048, 512), (2048, 512, 2048)):
        a2 = np.random.rand(M2, K2).astype(np.float32)
        b2 = np.random.rand(K2, N2).astype(np.float32)
        fl = 2.0 * M2 * N2 * K2
        gws = (int(np.ceil(N2 / 64)) * 8, int(np.ceil(M2 / 64)) * 8)
        cb2 = cl.Buffer(ctx, mf.WRITE_ONLY, M2 * N2 * 4)
        out = np.empty((M2, N2), np.float32)
        ref = a2 @ b2
        res = {}
        for tag, src, ty, kn in (('fp32', K32, 'float', 'v32'),
                                 ('fp16', K16, 'half', 'v16')):
            prog = build(ctx, f'{tag}_{M2}_{N2}_{K2}', src, K2, N2)
            aa = a2.astype(np.float16) if ty == 'half' else a2
            bb = b2.astype(np.float16) if ty == 'half' else b2
            buf = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=aa),
                   cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bb))
            kk = cl.Kernel(prog, kn)
            args = (np.int32(M2), np.int32(N2), np.int32(K2), buf[0], buf[1], cb2)
            kk(q, gws, (16, 16), *args)
            q.finish()
            cl.enqueue_copy(q, out, cb2).wait()
            e = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
            if e > 1e-2:
                print(f'   ⚠️ {tag} 在 {M2}×{N2}×{K2} 上误差 {e:.2e}，不计时')
                res[tag] = float('nan')
                continue
            t0 = time.perf_counter()
            for _ in range(5):
                kk(q, gws, (16, 16), *args)
            q.finish()
            res[tag] = fl / ((time.perf_counter() - t0) / 5) / 1e9
        r = res['fp16'] / res['fp32'] if res['fp32'] == res['fp32'] else float('nan')
        print(f'{M2:>8}×{N2:<5}×{K2:<4} {res["fp32"]:>12.1f} {res["fp16"]:>12.1f} '
              f'{r:>6.2f}×', flush=True)


if __name__ == '__main__':
    sys.exit(main())
