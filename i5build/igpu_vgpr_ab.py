"""用 -save-temps 拿到的 code object 属性，回答：fp16 那 1.9× 是不是占用率来的？

GCN 上每个 SIMD 的向量寄存器文件是 256 VGPR/lane。
一个 wave 用 N 个 VGPR ⇒ 每 SIMD 能常驻 floor(256/N) 个 wave。
所以 VGPR 数是"占用率"的直接读数：N=100 → 2 个 wave；N=150 → 只剩 1 个。

如果 fp16 版（half 存 A/B、fp32 累加）比 fp32 版 VGPR 明显少，那 1.9× 就解释通了，
而且下一步优化方向也就明确了：在 tile 不变的前提下压 VGPR。

⚠️ 本脚本只比【编译产物属性】，不跑性能 —— 所以不受训练任务污染。
"""
import glob
import os
import re
import shutil
import sys

import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '_vgreg')

TAIL32 = r'''
    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) {
            int cc = bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc] = acc[i][j];
        }
    }
}
'''

K32 = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
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
    const int AROW4 = TK/4, BCOL4 = TS/4;
    const int ATOT4 = TS*AROW4, BTOT4 = TK*BCOL4;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (ATOT4+BTOT4+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < ATOT4) {
                int row = f/AROW4, kq = f%AROW4;
                int gr = by+row, gc = k0+kq*4;
                float4 v = (float4)0.f;
                if (gr<M && gc+3<K) v = *(const __global float4*)(A+gr*K+gc);
                As[kq*4+0][row]=v.x; As[kq*4+1][row]=v.y; As[kq*4+2][row]=v.z; As[kq*4+3][row]=v.w;
            } else if (f < ATOT4+BTOT4) {
                int g = f-ATOT4, kk = g/BCOL4, cq = g%BCOL4;
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
''' + TAIL32

K16 = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
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
    const int AROW8 = TK/8, BCOL8 = TS/8;
    const int ATOT8 = TS*AROW8, BTOT8 = TK*BCOL8;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (ATOT8+BTOT8+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < ATOT8) {
                int row = f/AROW8, kq = f%AROW8;
                int gr = by+row, gc = k0+kq*8;
                half8 v = (half8)(0);
                if (gr<M && gc+7<K) v = *(const __global half8*)(A+gr*K+gc);
                As[kq*8+0][row]=v.s0; As[kq*8+1][row]=v.s1; As[kq*8+2][row]=v.s2; As[kq*8+3][row]=v.s3;
                As[kq*8+4][row]=v.s4; As[kq*8+5][row]=v.s5; As[kq*8+6][row]=v.s6; As[kq*8+7][row]=v.s7;
            } else if (f < ATOT8+BTOT8) {
                int g = f-ATOT8, kk = g/BCOL8, cq = g%BCOL8;
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
''' + TAIL32


def build_and_read(name, src):
    d = os.path.join(BASE, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        ctx = cl.Context([cl.get_platforms()[0].get_devices()[0]])
        prog = cl.Program(ctx, src)
        prog.build(options=['-save-temps', '-cl-fast-relaxed-math', '-cl-mad-enable'])
    finally:
        os.chdir(cwd)
    isa = glob.glob(os.path.join(d, '*.isa'))
    if not isa:
        print(f'{name}: 没有生成 .isa')
        return {}
    txt = open(isa[0], encoding='utf-8', errors='replace').read()
    keys = {
        'vgpr': r'workitem_vgpr_count\s*=\s*(\d+)',
        'sgpr': r'wavefront_sgpr_count\s*=\s*(\d+)',
        'lds': r'workgroup_group_segment_byte_size\s*=\s*(\d+)',
        'scratch': r'max_scratch_backing_memory_byte_size\s*=\s*(\d+)',
    }
    out = {}
    for k, pat in keys.items():
        m = re.search(pat, txt)
        out[k] = int(m.group(1)) if m else -1
    if out['vgpr'] > 0:
        out['waves_per_simd'] = 256 // out['vgpr']
        out['workgroups_per_cu'] = (out['waves_per_simd'] * 64) // 64   # 每工作组 2 wave
    return out


print(f'{"内核":<10} {"VGPR":>6} {"SGPR":>6} {"LDS(B)":>8} {"scratch":>8} '
      f'{"SIMD 可驻留 wave":>17} {"占用率估算":>11}')
res = {}
for name, src in (('fp32', K32), ('fp16op', K16)):
    r = build_and_read(name, src)
    res[name] = r
    if not r:
        continue
    waves = 256 // r['vgpr']
    occ = min(1.0, (waves * 64) / 640)     # GCN 每 CU 最多 40 wave（640 lanes）
    print(f'{name:<10} {r["vgpr"]:>6} {r["sgpr"]:>6} {r["lds"]:>8} {r["scratch"]:>8} '
          f'{waves:>17} {occ*100:>10.1f}%')

if 'fp32' in res and 'fp16op' in res and res['fp32'].get('vgpr', 0) > 0:
    a, b = res['fp32'], res['fp16op']
    print(f'\nVGPR: fp32 {a["vgpr"]} vs fp16 {b["vgpr"]}   '
          f'LDS: {a["lds"]} vs {b["lds"]} B')
