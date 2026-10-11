"""双缓冲预取：iGPU 探底里最后一个正统手段。

诊断链（前面几轮一步步排除出来的）：
    实测 FMA 峰值（16 条独立链）= 429 G-FMA/s ⇒ 真实 fp32 峰值 858 GFLOPS（≈1.12 GHz，
    而报告里"955~1150"是按 1.5 GHz 算的）
    我们的 GEMM = 313~361 GFLOPS = 峰值的 36~42%
    · 指令密度不是原因：float4 读 LDS 无收益（0.97~1.04×）
    · LDS 容量也不是直接原因：A 出 LDS 反而更慢（0.71~0.89×）
    ⇒ 剩下唯一站得住的解释：**全局访存延迟藏不住**
      （每轮 k-tile 要发 256 次全局读，延迟 400~600 周期，而只有 7 个工作组常驻）

双缓冲就是针对这个的标准解法：**算当前块的同时，把下一块预取到寄存器**，
算完再写进 LDS —— 全局延迟被 512 条 FMA 盖住。

代价必须一起报：LDS 翻倍（4352 → 8704 B）⇒ 常驻工作组 7 降 3。
所以这个实验真正测的是：**"延迟重叠"能不能赢过"常驻数减半"**。
"""
import os
import re
import shutil
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '_db')

COMMON = r'''
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

# ---------------- 基线：单缓冲 ----------------
K32 = COMMON + r'''
__kernel void g32(const int M, const int N, const int K,
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

# ---------------- 双缓冲：算当前块时把下一块预取到寄存器 ----------------
K32DB = COMMON + r'''
__kernel void g32db(const int M, const int N, const int K,
                    const __global float* restrict A,
                    const __global float* restrict B,
                    __global float* restrict C) {
    __local float As[2][TK][TS + 4];      // ← LDS 翻倍
    __local float Bs[2][TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int AROW = TK/4, BCOL = TS/4;

    // 每个线程固定负责 4 个 float4：A 的两个（f=lane, lane+64，ATOT=128），
    // B 的两个（g=lane, lane+64，BTOT=128）。全部线程都满，无分支。
    const int ar0 = lane/AROW,      ak0 = (lane%AROW)*4;
    const int ar1 = (lane+64)/AROW, ak1 = ((lane+64)%AROW)*4;
    const int bk0 = lane/BCOL,      bc0 = (lane%BCOL)*4;
    const int bk1 = (lane+64)/BCOL, bc1 = ((lane+64)%BCOL)*4;

    // ---- 序言：第 0 块 ----
    {
        float4 va0 = (float4)0.f, va1 = (float4)0.f, vb0 = (float4)0.f, vb1 = (float4)0.f;
        int gr = by+ar0, gc = ak0;
        if (gr<M && gc+3<K) va0 = *(const __global float4*)(A+gr*K+gc);
        gr = by+ar1; gc = ak1;
        if (gr<M && gc+3<K) va1 = *(const __global float4*)(A+gr*K+gc);
        gr = bk0; gc = bx+bc0;
        if (gr<K && gc+3<N) vb0 = *(const __global float4*)(B+gr*N+gc);
        gr = bk1; gc = bx+bc1;
        if (gr<K && gc+3<N) vb1 = *(const __global float4*)(B+gr*N+gc);
        As[0][ak0/4+0][ar0]=va0.x; As[0][ak0/4+1][ar0]=va0.y;
        As[0][ak0/4+2][ar0]=va0.z; As[0][ak0/4+3][ar0]=va0.w;
        As[0][ak1/4+0][ar1]=va1.x; As[0][ak1/4+1][ar1]=va1.y;
        As[0][ak1/4+2][ar1]=va1.z; As[0][ak1/4+3][ar1]=va1.w;
        Bs[0][bk0][bc0+0]=vb0.x; Bs[0][bk0][bc0+1]=vb0.y;
        Bs[0][bk0][bc0+2]=vb0.z; Bs[0][bk0][bc0+3]=vb0.w;
        Bs[0][bk1][bc1+0]=vb1.x; Bs[0][bk1][bc1+1]=vb1.y;
        Bs[0][bk1][bc1+2]=vb1.z; Bs[0][bk1][bc1+3]=vb1.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur = 0;
    for (int k0 = 0; k0 < K; k0 += TK) {
        const int nxt = cur ^ 1;
        const int k1 = k0 + TK;
        const int has = (k1 < K);

        // 1) 先把下一块的全局读【发出去】（写进寄存器，不落 LDS）
        float4 va0 = (float4)0.f, va1 = (float4)0.f, vb0 = (float4)0.f, vb1 = (float4)0.f;
        if (has) {
            int gr = by+ar0, gc = k1+ak0;
            if (gr<M && gc+3<K) va0 = *(const __global float4*)(A+gr*K+gc);
            gr = by+ar1; gc = k1+ak1;
            if (gr<M && gc+3<K) va1 = *(const __global float4*)(A+gr*K+gc);
            gr = k1+bk0; gc = bx+bc0;
            if (gr<K && gc+3<N) vb0 = *(const __global float4*)(B+gr*N+gc);
            gr = k1+bk1; gc = bx+bc1;
            if (gr<K && gc+3<N) vb1 = *(const __global float4*)(B+gr*N+gc);
        }

        // 2) 算当前块 —— 这 512 条 FMA 就是用来盖住上面那 4 次全局读的延迟
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i=0;i<TM;i++) a[i]=As[cur][k][ty*TM+i];
            for (int j=0;j<TN;j++) b[j]=Bs[cur][k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }

        // 3) 预取的数据这时应该已经到了，落进另一份缓冲
        if (has) {
            As[nxt][ak0/4+0][ar0]=va0.x; As[nxt][ak0/4+1][ar0]=va0.y;
            As[nxt][ak0/4+2][ar0]=va0.z; As[nxt][ak0/4+3][ar0]=va0.w;
            As[nxt][ak1/4+0][ar1]=va1.x; As[nxt][ak1/4+1][ar1]=va1.y;
            As[nxt][ak1/4+2][ar1]=va1.z; As[nxt][ak1/4+3][ar1]=va1.w;
            Bs[nxt][bk0][bc0+0]=vb0.x; Bs[nxt][bk0][bc0+1]=vb0.y;
            Bs[nxt][bk0][bc0+2]=vb0.z; Bs[nxt][bk0][bc0+3]=vb0.w;
            Bs[nxt][bk1][bc1+0]=vb1.x; Bs[nxt][bk1][bc1+1]=vb1.y;
            Bs[nxt][bk1][bc1+2]=vb1.z; Bs[nxt][bk1][bc1+3]=vb1.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        cur = nxt;
    }
''' + TAIL


def build(ctx, name, src, keep=False):
    d = os.path.join(BASE, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        prog = cl.Program(ctx, src).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable', '-save-temps'])
    finally:
        os.chdir(cwd)
    res = {}
    import glob
    sym = f'__OpenCL_{name}_kernel'
    for f in glob.glob(os.path.join(d, '*.isa')):
        txt = open(f, encoding='utf-8', errors='replace').read()
        for chunk in txt.split('AMD Kernel Code for &'):
            if not chunk.startswith(sym):
                continue
            for key, pat in (('vgpr', r'workitem_vgpr_count\s*=\s*(\d+)'),
                             ('lds', r'workgroup_group_segment_byte_size\s*=\s*(\d+)'),
                             ('scratch', r'max_scratch_backing_memory_byte_size\s*=\s*(\d+)')):
                m = re.search(pat, chunk)
                if m:
                    res[key] = int(m.group(1))
    return prog, res


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    print(f'device {dev.name} | {dev.max_compute_units} CU | LDS {dev.local_mem_size//1024} KB '
          f'（机器空闲）')

    progs = {}
    for name, src in (('g32', K32), ('g32db', K32DB)):
        p, res = build(ctx, name, src)
        progs[name] = p
        lds = res.get('lds', -1)
        print(f'{name:<8} LDS={lds:>5} B  ⇒ LDS 允许常驻 {32768//max(lds,1):>2} 个工作组  '
              f'VGPR={res.get("vgpr","?")} scratch={res.get("scratch","?")}')

    mf = cl.mem_flags
    shapes = [(1024, 1024, 1024, '锚点'), (2048, 512, 512, 'AR 注意力'),
              (2048, 2048, 512, 'AR FFN up'), (2048, 512, 2048, 'AR FFN down')]
    print(f"\n{'形状':>20} {'g32':>9} {'g32db':>9} {'db/单缓冲':>10}  说明")
    for M, N, K, note in shapes:
        a = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(K, N).astype(np.float32)
        fl = 2.0 * M * N * K
        gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
        ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
              cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
        cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        out = np.empty((M, N), np.float32)
        res = {}
        errs = {}
        for name in ('g32', 'g32db'):
            k = cl.Kernel(progs[name], name)
            args = (np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cbuf)
            k(q, gws, (8, 8), *args)
            q.finish()
            t0 = time.perf_counter()
            for _ in range(3):
                k(q, gws, (8, 8), *args)
            q.finish()
            res[name] = fl / ((time.perf_counter() - t0) / 3) / 1e9
            cl.enqueue_copy(q, out, cbuf).wait()
            ref = a @ b
            errs[name] = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        flag = '' if errs['g32db'] < 1e-4 else f"  ⚠️ err={errs['g32db']:.1e}"
        print(f'{M:>8}×{N:<5}×{K:<4} {res["g32"]:>9.1f} {res["g32db"]:>9.1f} '
              f'{res["g32db"]/res["g32"]:>9.2f}×  {note}{flag}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
