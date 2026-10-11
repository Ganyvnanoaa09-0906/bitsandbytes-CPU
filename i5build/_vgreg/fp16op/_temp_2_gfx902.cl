/* Compiler options:
-save-temps -cl-fast-relaxed-math -cl-mad-enable -I C:\Users\Gan\AppData\Local\Programs\Python\Python311\Lib\site-packages\pyopencl\cl -D__AMD__=1 -D__gfx902__=1 -D__gfx902=1 -DFP_FAST_FMAF=1 -DFP_FAST_FMA=1 -D__IMAGE_SUPPORT__=1 -cl-denorms-are-zero -m64 -Dcl_khr_fp64=1 -Dcl_amd_fp64=1 -Dcl_khr_global_int32_base_atomics=1 -Dcl_khr_global_int32_extended_atomics=1 -Dcl_khr_local_int32_base_atomics=1 -Dcl_khr_local_int32_extended_atomics=1 -Dcl_khr_int64_base_atomics=1 -Dcl_khr_int64_extended_atomics=1 -Dcl_khr_3d_image_writes=1 -Dcl_khr_byte_addressable_store=1 -Dcl_khr_fp16=1 -Dcl_khr_gl_sharing=1 -Dcl_khr_gl_depth_images=1 -Dcl_amd_device_attribute_query=1 -Dcl_amd_vec3=1 -Dcl_amd_printf=1 -Dcl_amd_media_ops=1 -Dcl_amd_media_ops2=1 -Dcl_amd_popcnt=1 -Dcl_khr_d3d10_sharing=1 -Dcl_khr_d3d11_sharing=1 -Dcl_khr_dx9_media_sharing=1 -Dcl_khr_image2d_from_buffer=1 -Dcl_khr_subgroups=1 -Dcl_khr_gl_event=1 -Dcl_khr_depth_images=1 -Dcl_khr_mipmap_image=1 -Dcl_khr_mipmap_image_writes=1 -Dcl_amd_liquid_flash=1 -Dcl_amd_copy_buffer_p2p=1 -Dcl_amd_planar_yuv=1 
*/


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

    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) {
            int cc = bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc] = acc[i][j];
        }
    }
}


__constant int pyopencl_defeat_cache_0a82fd13bfd64bca99cdc9ac5edd7e8c = 0;