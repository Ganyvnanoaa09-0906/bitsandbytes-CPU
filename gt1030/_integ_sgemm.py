# -*- coding: utf-8 -*-
"""把自写 k=64 batched GEMM 接进 ar_v2_multi：S = Qn·Knᵀ（前向）与 dP = dO·Vbᵀ（反向）。

独占口径实测：128x64 t8x4（16x16 线程）8.40ms vs CLBlast 14.98ms = **1.78×**，数值 rel 2.9e-3。
每层省 9.9ms(S) + 6.8ms(dP) ≈ 16.7ms ⇒ 10 层约 167ms/step。
alpha 固定 1.0，softmax 里的 scale 不动 —— 这样原来验证过的路径语义完全不变。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# ---- 1) 新 kernel ----
s = s.replace("""__kernel void scale_half(__global half* g, float f, int n) {""",
"""// k=64 的 batched GEMM：C[b] = A[b]·B[b]ᵀ（A/B 都是 (T,64) row-major）
// 用于 S = Qn·Knᵀ 与 dP = dO·Vbᵀ。128x64 tile / 每线程 8x4 / 16x16 线程 = 实测 1.78x CLBlast。
__kernel void sgemm_nt64(__global const half* A, __global const half* B,
                         __global half* C, int T, float alpha) {
    __local half At[64 * 128];
    __local half Bt[64 * 64];
    int bh = get_group_id(2);
    int ti = get_group_id(0), tj = get_group_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * 16 + lx;
    __global const half* Ap = A + (long)bh * T * 64;
    __global const half* Bp = B + (long)bh * T * 64;
    __global half* Cp = C + (long)bh * T * T;
    int mb = ti * 128, nb = tj * 64;
    for (int t = tid; t < 128 * 16; t += 256) {
        int r = t / 16, c4 = (t % 16) * 4;
        half4 va = vload4(0, Ap + (long)(mb + r) * 64 + c4);
        At[(c4 + 0) * 128 + r] = va.x; At[(c4 + 1) * 128 + r] = va.y;
        At[(c4 + 2) * 128 + r] = va.z; At[(c4 + 3) * 128 + r] = va.w;
    }
    for (int t = tid; t < 64 * 16; t += 256) {
        int r = t / 16, c4 = (t % 16) * 4;
        half4 vb = vload4(0, Bp + (long)(nb + r) * 64 + c4);
        Bt[(c4 + 0) * 64 + r] = vb.x; Bt[(c4 + 1) * 64 + r] = vb.y;
        Bt[(c4 + 2) * 64 + r] = vb.z; Bt[(c4 + 3) * 64 + r] = vb.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int m0 = ly * 8, n0 = lx * 4;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int kk = 0; kk < 64; ++kk) {
        float av[8], bv[4];
        half4 a0 = vload4(0, (__local half*)At + kk * 128 + m0);
        half4 a1 = vload4(0, (__local half*)At + kk * 128 + m0 + 4);
        av[0] = (float)a0.x; av[1] = (float)a0.y; av[2] = (float)a0.z; av[3] = (float)a0.w;
        av[4] = (float)a1.x; av[5] = (float)a1.y; av[6] = (float)a1.z; av[7] = (float)a1.w;
        half4 b0 = vload4(0, (__local half*)Bt + kk * 64 + n0);
        bv[0] = (float)b0.x; bv[1] = (float)b0.y; bv[2] = (float)b0.z; bv[3] = (float)b0.w;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
    }
    for (int i = 0; i < 8; ++i) {
        int gm = mb + m0 + i;
        if (gm >= T) break;
        for (int j = 0; j < 4; ++j) {
            int gn = nb + n0 + j;
            if (gn >= T) break;
            Cp[(long)gm * T + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
__kernel void scale_half(__global half* g, float f, int n) {""")
s = s.replace("""          'ce_softmax', 'scale_half', 'scale_float', 'rms_dw_from_ss', 'col_sum_prod')""",
              """          'ce_softmax', 'scale_half', 'scale_float', 'rms_dw_from_ss', 'col_sum_prod',
          'sgemm_nt64')""")

# ---- 2) 前向 S ----
s = s.replace("""        A.hgemm_batched(b['Qn'], b['Kn'], b['S'], T, T, HD, A.NO, A.YES,
                        HD, HD, T, T * HD, T * HD, T * T)""",
"""        # S = Qn·Knᵀ：自写 k=64 batched（比 CLBlast strided-batched 快 1.78×，独占口径）
        K['sgemm_nt64'](d.q, ((T + 127) // 128 * 16, (T + 63) // 64 * 16, BH), (16, 16, 1),
                        b['Qn'], b['Kn'], b['S'], np.int32(T), np.float32(1.0))""")
# ---- 3) 反向 dP ----
s = s.replace("""        A.hgemm_batched(b['dO'], b['Vb'], b['dP'], T, T, HD, A.NO, A.YES,
                        HD, HD, T, T * HD, T * HD, T * T)""",
"""        K['sgemm_nt64'](d.q, ((T + 127) // 128 * 16, (T + 63) // 64 * 16, BH), (16, 16, 1),
                        b['dO'], b['Vb'], b['dP'], np.int32(T), np.float32(1.0))""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('integrated sgemm_nt64 (S and dP)')
