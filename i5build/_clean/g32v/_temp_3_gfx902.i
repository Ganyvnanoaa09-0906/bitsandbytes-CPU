# 1 "C:\\Users\\Gan\\AppData\\Local\\Temp\\\\OCL19148T3.cl"
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
# 1 "C:\\Users\\Gan\\AppData\\Local\\Temp\\\\OCL19148T3.cl" 2

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
__kernel void g32v(const int M, const int N, const int K,
                     const __global float* restrict A,
                     const __global float* restrict B,
                     __global float* restrict C) {
    __local float As[8][64 + 4];
    __local float Bs[8][64 + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * 64, by = get_group_id(1) * 64;
    const int lane = ty * 8 + tx;
    float acc[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
    const int VEC = 4;
    const int AROW = 8/VEC, BCOL = 64/VEC;
    const int ATOT = 64*AROW, BTOT = 8*BCOL;
    for (int k0 = 0; k0 < K; k0 += 8) {
        for (int p = 0; p < (ATOT+BTOT+64 -1)/64; p++) {
            int f = p*64 + lane;
            if (f < ATOT) {
                int row = f/AROW, kq = f%AROW;
                int gr = by+row, gc = k0+kq*VEC;
                float4 v = (float4)(0);
                if (gr<M && gc+VEC-1<K) v = *(const __global float4*)(A+gr*K+gc);
                As[kq*VEC+0][row]=v.s0; As[kq*VEC+1][row]=v.s1;
                As[kq*VEC+2][row]=v.s2; As[kq*VEC+3][row]=v.s3;

            } else if (f < ATOT+BTOT) {
                int g = f-ATOT, kk = g/BCOL, cq = g%BCOL;
                int gr = k0+kk, gc = bx+cq*VEC;
                float4 v = (float4)(0);
                if (gr<K && gc+VEC-1<N) v = *(const __global float4*)(B+gr*N+gc);
                Bs[kk][cq*VEC+0]=v.s0; Bs[kk][cq*VEC+1]=v.s1;
                Bs[kk][cq*VEC+2]=v.s2; Bs[kk][cq*VEC+3]=v.s3;

            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 8; k++) {
            float4 a0 = *(const __local float4*)(&As[k][ty*8]);
            float4 a1 = *(const __local float4*)(&As[k][ty*8 + 4]);
            float4 b0 = *(const __local float4*)(&Bs[k][tx*8]);
            float4 b1 = *(const __local float4*)(&Bs[k][tx*8 + 4]);
            float av[8] = {a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w};
            float bv[8] = {b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
            for (int i=0;i<8;i++)
                for (int j=0;j<8;j++) acc[i][j]=fma(av[i],bv[j],acc[i][j]);
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



__constant int pyopencl_defeat_cache_ecbe35aa71d749fa86e1e63e83799ef4 = 0;
