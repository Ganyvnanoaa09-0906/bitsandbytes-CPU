# 1 "C:\\Users\\Gan\\AppData\\Local\\Temp\\\\OCL19148T4.cl"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
#define __clang__ 1
#define __clang_major__ 3
#define __clang_minor__ 9
#define __ENDIAN_LITTLE__ 1
#define __SPIR64 1
#define __SPIR64__ 1
#define __STDC__ 1
#define __STDC_HOSTED__ 1
#define __STDC_VERSION__ 199901L
#define __STDC_UTF_16__ 1
#define __STDC_UTF_32__ 1
#define __OPENCL_C_VERSION__ 120
#define __OPENCL_VERSION__ 120

# 1 "<command line>" 1
#define __FAST_RELAXED_MATH__ 1
#define __AMD__ 1
#define __gfx902__ 1
#define __gfx902 1
#define FP_FAST_FMAF 1
#define FP_FAST_FMA 1
#define __IMAGE_SUPPORT__ 1
#define cl_khr_fp64 1
#define cl_amd_fp64 1
#define cl_khr_global_int32_base_atomics 1
#define cl_khr_global_int32_extended_atomics 1
#define cl_khr_local_int32_base_atomics 1
#define cl_khr_local_int32_extended_atomics 1
#define cl_khr_int64_base_atomics 1
#define cl_khr_int64_extended_atomics 1
#define cl_khr_3d_image_writes 1
#define cl_khr_byte_addressable_store 1
#define cl_khr_fp16 1
#define cl_khr_gl_sharing 1
#define cl_khr_gl_depth_images 1
#define cl_amd_device_attribute_query 1
#define cl_amd_vec3 1
#define cl_amd_printf 1
#define cl_amd_media_ops 1
#define cl_amd_media_ops2 1
#define cl_amd_popcnt 1
#define cl_khr_d3d10_sharing 1
#define cl_khr_d3d11_sharing 1
#define cl_khr_dx9_media_sharing 1
#define cl_khr_image2d_from_buffer 1
#define cl_khr_subgroups 1
#define cl_khr_gl_event 1
#define cl_khr_depth_images 1
#define cl_khr_mipmap_image 1
#define cl_khr_mipmap_image_writes 1
#define cl_amd_liquid_flash 1
#define cl_amd_copy_buffer_p2p 1
#define cl_amd_planar_yuv 1
#define __IMAGE_SUPPORT__ 1
#define __AMD__ 1
# 1 "<built-in>" 2
# 1 "C:\\Users\\Gan\\AppData\\Local\\Temp\\\\OCL19148T4.cl" 2

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
__kernel void g16(const int M, const int N, const int K,
                     const __global half* restrict A,
                     const __global half* restrict B,
                     __global float* restrict C) {
    __local half As[8][64 + 8];
    __local half Bs[8][64 + 8];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * 64, by = get_group_id(1) * 64;
    const int lane = ty * 8 + tx;
    float acc[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
    const int VEC = 8;
    const int AROW = 8/VEC, BCOL = 64/VEC;
    const int ATOT = 64*AROW, BTOT = 8*BCOL;
    for (int k0 = 0; k0 < K; k0 += 8) {
        for (int p = 0; p < (ATOT+BTOT+64 -1)/64; p++) {
            int f = p*64 + lane;
            if (f < ATOT) {
                int row = f/AROW, kq = f%AROW;
                int gr = by+row, gc = k0+kq*VEC;
                half8 v = (half8)(0);
                if (gr<M && gc+VEC-1<K) v = *(const __global half8*)(A+gr*K+gc);
                As[kq*VEC+0][row]=v.s0; As[kq*VEC+1][row]=v.s1;
                As[kq*VEC+2][row]=v.s2; As[kq*VEC+3][row]=v.s3;
                As[kq*8+4][row]=v.s4; As[kq*8+5][row]=v.s5; As[kq*8+6][row]=v.s6; As[kq*8+7][row]=v.s7;
            } else if (f < ATOT+BTOT) {
                int g = f-ATOT, kk = g/BCOL, cq = g%BCOL;
                int gr = k0+kk, gc = bx+cq*VEC;
                half8 v = (half8)(0);
                if (gr<K && gc+VEC-1<N) v = *(const __global half8*)(B+gr*N+gc);
                Bs[kk][cq*VEC+0]=v.s0; Bs[kk][cq*VEC+1]=v.s1;
                Bs[kk][cq*VEC+2]=v.s2; Bs[kk][cq*VEC+3]=v.s3;
                Bs[kk*1][cq*8+4]=v.s4;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 8; k++) {
            float a[8], b[8];
            for (int i=0;i<8;i++) a[i]=(float)As[k][ty*8 +i];
            for (int j=0;j<8;j++) b[j]=(float)Bs[k][tx*8 +j];
            for (int i=0;i<8;i++)
                for (int j=0;j<8;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i=0;i<8;i++) {
        int rc = by+ty*8 +i; if (rc>=M) continue;
        for (int j=0;j<8;j++) {
            int cc = bx+tx*8 +j; if (cc>=N) continue;
            C[rc*N+cc] = acc[i][j];
        }
    }
}



__constant int pyopencl_defeat_cache_7bbb5060a4de4d058c2adbc2af6afefa = 0;
