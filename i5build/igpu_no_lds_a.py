"""把 A 踢出 LDS（显式四份源码，不做字符串魔法）。

推理链（上一轮用 -save-temps 读出来的）：
    每 CU 32 KB LDS，一个工作组(8x8=64线程) = 1 个 wave
    fp32    A+B 进 LDS：4352 B ⇒ 32768/4352 =  7 个 wave 常驻（上限 40）
    fp16    A+B 进 LDS：2304 B ⇒ 32768/2304 = 14 个
⇒ LDS 是墙。fp16 那 1.9× 就是 LDS 减半换来的，但要改整条精度链路。

同样减半 LDS 且【不动精度】的办法：A 不进 LDS。A 的每个元素在 tile 内被复用
TS/TM=8 次，而这 8 个读取者的地址【完全相同】（索引与 tx 无关、同在一个 wave）
⇒ L1 广播即可，不必占 LDS。

同时必须报准确的 VGPR：若每线程 ~120 个，每 SIMD 只放 2 个 wave（8 个/CU），
那 LDS 松到 2176 也没用。上一版 31/29 是解析 bug（抓到辅助函数），这版按 kernel
名字切块。
"""
import glob
import os
import re
import shutil
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '_nla')

PROLOG = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
'''

TAIL = r'''
    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) {
            int cc = bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc] = acc[i][j];
        }
    }
}
'''

# ---------------- 1) fp32，A+B 都进 LDS（基线） ----------------
K32 = PROLOG + r'''
__kernel void gemm32(const int M, const int N, const int K,
                     const __global float* restrict A,
                     const __global float* restrict B,
                     __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int AROW = TK/4, BCOL = TS/4;
    const int ATOT = TS*AROW, BTOT = TK*BCOL;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (ATOT+BTOT+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < ATOT) {
                int row = f/AROW, kq = f%AROW;
                int gr = by+row, gc = k0+kq*4;
                float4 v = (float4)0.f;
                if (gr<M && gc+3<K) v = *(const __global float4*)(A+gr*K+gc);
                As[kq*4+0][row]=v.x; As[kq*4+1][row]=v.y; As[kq*4+2][row]=v.z; As[kq*4+3][row]=v.w;
            } else if (f < ATOT+BTOT) {
                int g = f-ATOT, kk = g/BCOL, cq = g%BCOL;
                int gr = k0+kk, gc = bx+cq*4;
                float4 v = (float4)0.f;
                if (gr<K && gc+3<N) v = *(const __global float4*)(B+gr*N+gc);
                Bs[kk][cq*4+0]=v.x; Bs[kk][cq*4+1]=v.y; Bs[kk][cq*4+2]=v.z; Bs[kk][cq*4+3]=v.w;
            }
        }
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
''' + TAIL

# ---------------- 2) fp32，只有 B 进 LDS（A 走 L1） ----------------
K32NL = PROLOG + r'''
__kernel void gemm32nl(const int M, const int N, const int K,
                       const __global float* restrict A,
                       const __global float* restrict B,
                       __global float* restrict C) {
    __local float Bs[TK][TS + 4];              // ← 只有 B 占 LDS
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int BCOL = TS/4;
    const int BTOT = TK*BCOL;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (BTOT+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < BTOT) {
                int kk = f/BCOL, cq = f%BCOL;
                int gr = k0+kk, gc = bx+cq*4;
                float4 v = (float4)0.f;
                if (gr<K && gc+3<N) v = *(const __global float4*)(B+gr*N+gc);
                Bs[kk][cq*4+0]=v.x; Bs[kk][cq*4+1]=v.y; Bs[kk][cq*4+2]=v.z; Bs[kk][cq*4+3]=v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            // A 直接从 global 读：8 个 tx 线程读同一地址 ⇒ L1 广播
            for (int i=0;i<TM;i++) a[i]=A[(by+ty*TM+i)*K + (k0+k)];
            for (int j=0;j<TN;j++) b[j]=Bs[k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
''' + TAIL

# ---------------- 3) fp16 操作数，A+B 都进 LDS ----------------
K16 = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n' + PROLOG + r'''
__kernel void gemm16(const int M, const int N, const int K,
                     const __global half* restrict A,
                     const __global half* restrict B,
                     __global float* restrict C) {
    __local half As[TK][TS + 8];
    __local half Bs[TK][TS + 8];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int AROW = TK/8, BCOL = TS/8;
    const int ATOT = TS*AROW, BTOT = TK*BCOL;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (ATOT+BTOT+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < ATOT) {
                int row = f/AROW, kq = f%AROW;
                int gr = by+row, gc = k0+kq*8;
                half8 v = (half8)(0);
                if (gr<M && gc+7<K) v = *(const __global half8*)(A+gr*K+gc);
                As[kq*8+0][row]=v.s0; As[kq*8+1][row]=v.s1; As[kq*8+2][row]=v.s2; As[kq*8+3][row]=v.s3;
                As[kq*8+4][row]=v.s4; As[kq*8+5][row]=v.s5; As[kq*8+6][row]=v.s6; As[kq*8+7][row]=v.s7;
            } else if (f < ATOT+BTOT) {
                int g = f-ATOT, kk = g/BCOL, cq = g%BCOL;
                int gr = k0+kk, gc = bx+cq*8;
                half8 v = (half8)(0);
                if (gr<K && gc+7<N) v = *(const __global half8*)(B+gr*N+gc);
                Bs[kk][cq*8+0]=v.s0; Bs[kk][cq*8+1]=v.s1; Bs[kk][cq*8+2]=v.s2; Bs[kk][cq*8+3]=v.s3;
                Bs[kk][cq*8+4]=v.s4; Bs[kk][cq*8+5]=v.s5; Bs[kk][cq*8+6]=v.s6; Bs[kk][cq*8+7]=v.s7;
            }
        }
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
''' + TAIL

# ---------------- 4) fp16 操作数 + A 不进 LDS（LDS 最小） ----------------
K16NL = '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n' + PROLOG + r'''
__kernel void gemm16nl(const int M, const int N, const int K,
                       const __global half* restrict A,
                       const __global half* restrict B,
                       __global float* restrict C) {
    __local half Bs[TK][TS + 8];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int BCOL = TS/8;
    const int BTOT = TK*BCOL;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (BTOT+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < BTOT) {
                int kk = f/BCOL, cq = f%BCOL;
                int gr = k0+kk, gc = bx+cq*8;
                half8 v = (half8)(0);
                if (gr<K && gc+7<N) v = *(const __global half8*)(B+gr*N+gc);
                Bs[kk][cq*8+0]=v.s0; Bs[kk][cq*8+1]=v.s1; Bs[kk][cq*8+2]=v.s2; Bs[kk][cq*8+3]=v.s3;
                Bs[kk][cq*8+4]=v.s4; Bs[kk][cq*8+5]=v.s5; Bs[kk][cq*8+6]=v.s6; Bs[kk][cq*8+7]=v.s7;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i=0;i<TM;i++) a[i]=(float)A[(by+ty*TM+i)*K + (k0+k)];
            for (int j=0;j<TN;j++) b[j]=(float)Bs[k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
''' + TAIL

KERNELS = [('gemm32', K32, 'float'), ('gemm32nl', K32NL, 'float'),
           ('gemm16', K16, 'half'), ('gemm16nl', K16NL, 'half')]


def build(ctx, name, src):
    d = os.path.join(BASE, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        prog = cl.Program(ctx, src).build(
            options=['-save-temps', '-cl-fast-relaxed-math', '-cl-mad-enable'])
    finally:
        os.chdir(cwd)
    res = {}
    # ⚠️ dump 里的符号名带前后缀：__OpenCL_<name>_kernel（第一版按 <name> 匹配，
    # 四个全是"解析失败"）
    sym = f'__OpenCL_{name}_kernel'
    for f in glob.glob(os.path.join(d, '*.isa')):
        txt = open(f, encoding='utf-8', errors='replace').read()
        for chunk in txt.split('AMD Kernel Code for &'):
            if not chunk.startswith(sym):
                continue
            for key, pat in (('vgpr', r'workitem_vgpr_count\s*=\s*(\d+)'),
                             ('sgpr', r'wavefront_sgpr_count\s*=\s*(\d+)'),
                             ('lds', r'workgroup_group_segment_byte_size\s*=\s*(\d+)'),
                             ('scratch', r'max_scratch_backing_memory_byte_size\s*=\s*(\d+)')):
                m = re.search(pat, chunk)
                if m:
                    res[key] = int(m.group(1))
    return prog, res


def main():
    plat = cl.get_platforms()[0]
    dev = plat.get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    print(f'device {dev.name} | {dev.max_compute_units} CU | '
          f'LDS {dev.local_mem_size//1024} KB')

    progs, info = {}, {}
    print(f"\n{'变体':<10} {'VGPR':>5} {'SGPR':>5} {'LDS(B)':>7} {'scratch':>8} "
          f"{'LDS允wave':>10} {'VGPR允wave':>11} {'实际可驻留':>10}")
    for name, src, _ty in KERNELS:
        prog, res = build(ctx, name, src)
        progs[name] = prog
        info[name] = res
        if 'vgpr' not in res or 'lds' not in res:
            print(f'{name:<10}  解析失败')
            continue
        w_lds = 32768 // max(res['lds'], 1)
        w_vgpr = (256 // res['vgpr']) * 4
        res['waves'] = min(w_lds, w_vgpr, 40)
        print(f'{name:<10} {res["vgpr"]:>5} {res["sgpr"]:>5} {res["lds"]:>7} '
              f'{res.get("scratch", -1):>8} {w_lds:>10} {w_vgpr:>11} '
              f'{res["waves"]:>10}')

    print(f"\n{'形状':>20} " + ' '.join(f'{n:>10}' for n, _, _ in KERNELS))
    mf = cl.mem_flags
    for M, N, K in ((1024, 1024, 1024), (2048, 512, 512), (2048, 2048, 512)):
        a32 = np.random.rand(M, K).astype(np.float32)
        b32 = np.random.rand(K, N).astype(np.float32)
        fl = 2.0 * M * N * K
        gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
        bufs = {
            'float': (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a32),
                      cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b32)),
            'half': (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=a32.astype(np.float16)),
                     cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=b32.astype(np.float16))),
        }
        cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        row = []
        ref = None
        for name, _src, ty in KERNELS:
            k = cl.Kernel(progs[name], name)
            ab = bufs[ty]
            args = (np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cbuf)
            k(q, gws, (8, 8), *args)
            q.finish()
            t0 = time.perf_counter()
            for _ in range(3):
                k(q, gws, (8, 8), *args)
            q.finish()
            dt = (time.perf_counter() - t0) / 3
            # 正确性
            o = np.empty((M, N), np.float32)
            cl.enqueue_copy(q, o, cbuf).wait()
            a_ref = a32.astype(np.float16).astype(np.float32) if ty == 'half' else a32
            b_ref = b32.astype(np.float16).astype(np.float32) if ty == 'half' else b32
            r = a_ref @ b_ref
            err = float(np.abs(o - r).max() / (np.abs(r).max() + 1e-9))
            row.append(f'{fl/dt/1e9:>10.1f}' if err < 1e-4 else f'{"ERR":>10}')
        print(f'{M:>8}×{N:<5}×{K:<5} ' + ' '.join(row) +
              f'   nl/base={float(row[1])/float(row[0]):.2f}x '
              f'{float(row[3])/float(row[2]):.2f}x', flush=True)


if __name__ == '__main__':
    sys.exit(main())
