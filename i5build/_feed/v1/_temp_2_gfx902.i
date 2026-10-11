# 1 "C:\\Users\\Gan\\AppData\\Local\\Temp\\\\OCL15288T2.cl"
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
#define KC 512
#define NC 512
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
# 1 "C:\\Users\\Gan\\AppData\\Local\\Temp\\\\OCL15288T2.cl" 2

#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64

__kernel void v1(const int M, const int N, const int K,
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
    const int AROW = 8/4, BCOL = 64/4;


    const int ar0 = lane/AROW, ak0 = (lane%AROW)*4;
    const int ar1 = (lane+64)/AROW, ak1 = ((lane+64)%AROW)*4;
    const int bk0 = lane/BCOL, bc0 = (lane%BCOL)*4;
    const int bk1 = (lane+64)/BCOL, bc1 = ((lane+64)%BCOL)*4;
    const __global float* Ap0 = A + (by+ar0)*512 + ak0;
    const __global float* Ap1 = A + (by+ar1)*512 + ak1;
    const __global float* Bp0 = B + bk0*512 + bx + bc0;
    const __global float* Bp1 = B + bk1*512 + bx + bc1;

    for (int k0 = 0; k0 < K; k0 += 8) {
        float4 va0 = *(const __global float4*)(Ap0 + k0);
        float4 va1 = *(const __global float4*)(Ap1 + k0);
        float4 vb0 = *(const __global float4*)(Bp0 + k0*512);
        float4 vb1 = *(const __global float4*)(Bp1 + k0*512);
        As[ak0/4+0][ar0]=va0.x; As[ak0/4+1][ar0]=va0.y;
        As[ak0/4+2][ar0]=va0.z; As[ak0/4+3][ar0]=va0.w;
        As[ak1/4+0][ar1]=va1.x; As[ak1/4+1][ar1]=va1.y;
        As[ak1/4+2][ar1]=va1.z; As[ak1/4+3][ar1]=va1.w;
        Bs[bk0][bc0+0]=vb0.x; Bs[bk0][bc0+1]=vb0.y;
        Bs[bk0][bc0+2]=vb0.z; Bs[bk0][bc0+3]=vb0.w;
        Bs[bk1][bc1+0]=vb1.x; Bs[bk1][bc1+1]=vb1.y;
        Bs[bk1][bc1+2]=vb1.z; Bs[bk1][bc1+3]=vb1.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 8; k++) {
            float a[8], b[8];
            for (int i=0;i<8;i++) a[i]=As[k][ty*8 +i];
            for (int j=0;j<8;j++) b[j]=Bs[k][tx*8 +j];
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


__constant int pyopencl_defeat_cache_e9e02727ef2b4e65ae0ed3c348a07061 = 0;
