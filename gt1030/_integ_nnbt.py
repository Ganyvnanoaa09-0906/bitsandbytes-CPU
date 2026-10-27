# -*- coding: utf-8 -*-
"""把自写 batched nn GEMM 接进 ar_v2_multi：O = P·Vb（前向）与 dQ = dS·Kn（反向）。

独占口径：CLBlast 9.82ms(219 GFLOPS) vs 自写 6.54ms(328 GFLOPS) = 1.50×，rel 1.24e-2。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

KERN = r"""
// C[b] = A[b]·B[b]，A (T,T) row-major、B (T,64) row-major、C (T,64)。
// 用于 O = P·Vb 与 dQ = dS·Kn。MS=128/NS=64/BK=64，1.50x CLBlast。
__kernel void sgemm_nn_bt(__global const half* A, __global const half* B,
                          __global half* C, int T, float alpha) {
    __local half At[64 * 128];
    __local half Bt[64 * 64];
    int bh = get_group_id(2);
    int mb = get_group_id(0) * 128;
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * 16 + lx;
    __global const half* Ap = A + (long)bh * T * T;
    __global const half* Bp = B + (long)bh * T * 64;
    __global half* Cp = C + (long)bh * T * 64;
    int m0 = ly * 8, n0 = lx * 4;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < T; k0 += 64) {
        for (int t = tid; t < 128 * 16; t += 256) {
            int r = t / 16, c4 = (t % 16) * 4;
            half4 v = vload4(0, Ap + (long)(mb + r) * T + k0 + c4);
            At[(c4 + 0) * 128 + r] = v.x; At[(c4 + 1) * 128 + r] = v.y;
            At[(c4 + 2) * 128 + r] = v.z; At[(c4 + 3) * 128 + r] = v.w;
        }
        for (int t = tid; t < 64 * 16; t += 256) {
            int r = t / 16, c4 = (t % 16) * 4;
            half4 v = vload4(0, Bp + (long)(k0 + r) * 64 + c4);
            vstore4(v, 0, (__local half*)Bt + r * 64 + c4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 64; ++kk) {
            float av[8], bv[4];
            half4 a0 = vload4(0, (__local half*)At + kk * 128 + m0);
            half4 a1 = vload4(0, (__local half*)At + kk * 128 + m0 + 4);
            av[0] = (float)a0.x; av[1] = (float)a0.y; av[2] = (float)a0.z; av[3] = (float)a0.w;
            av[4] = (float)a1.x; av[5] = (float)a1.y; av[6] = (float)a1.z; av[7] = (float)a1.w;
            half4 b = vload4(0, (__local half*)Bt + kk * 64 + n0);
            bv[0] = (float)b.x; bv[1] = (float)b.y; bv[2] = (float)b.z; bv[3] = (float)b.w;
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < 8; ++i) {
        int gm = mb + m0 + i;
        if (gm >= T) break;
        for (int j = 0; j < 4; ++j) {
            int gn = n0 + j;
            if (gn >= 64) break;
            Cp[(long)gm * 64 + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
"""
s = s.replace("""__kernel void scale_half(__global half* g, float f, int n) {""",
              KERN + """__kernel void scale_half(__global half* g, float f, int n) {""")
s = s.replace("""          'sgemm_nt64')""", """          'sgemm_nt64', 'sgemm_nn_bt')""")

# forward: O = P·Vb
s = s.replace("""        A.hgemm_batched(b['P'], b['Vb'], b['O'], T, HD, T, A.NO, A.NO,
                        T, HD, HD, T * T, T * HD, T * HD)""",
"""        K['sgemm_nn_bt'](d.q, ((T + 127) // 128 * 16, 16, BH), (16, 16, 1),
                         b['P'], b['Vb'], b['O'], np.int32(T), np.float32(1.0))""")
# backward: dQ = dS·Kn
s = s.replace("""        A.hgemm_batched(b['dS'], b['Kn'], b['dQ'], T, HD, T, A.NO, A.NO,
                        T, HD, HD, T * T, T * HD, T * HD)""",
"""        K['sgemm_nn_bt'](d.q, ((T + 127) // 128 * 16, 16, BH), (16, 16, 1),
                         b['dS'], b['Kn'], b['dQ'], np.int32(T), np.float32(1.0))""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('integrated sgemm_nn_bt (O and dQ)')
