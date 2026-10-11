/* Compiler options:
-save-temps -cl-fast-relaxed-math -cl-mad-enable -I C:\Users\Gan\AppData\Local\Programs\Python\Python311\Lib\site-packages\pyopencl\cl -D__AMD__=1 -D__gfx902__=1 -D__gfx902=1 -DFP_FAST_FMAF=1 -DFP_FAST_FMA=1 -D__IMAGE_SUPPORT__=1 -cl-denorms-are-zero -m64 -Dcl_khr_fp64=1 -Dcl_amd_fp64=1 -Dcl_khr_global_int32_base_atomics=1 -Dcl_khr_global_int32_extended_atomics=1 -Dcl_khr_local_int32_base_atomics=1 -Dcl_khr_local_int32_extended_atomics=1 -Dcl_khr_int64_base_atomics=1 -Dcl_khr_int64_extended_atomics=1 -Dcl_khr_3d_image_writes=1 -Dcl_khr_byte_addressable_store=1 -Dcl_khr_fp16=1 -Dcl_khr_gl_sharing=1 -Dcl_khr_gl_depth_images=1 -Dcl_amd_device_attribute_query=1 -Dcl_amd_vec3=1 -Dcl_amd_printf=1 -Dcl_amd_media_ops=1 -Dcl_amd_media_ops2=1 -Dcl_amd_popcnt=1 -Dcl_khr_d3d10_sharing=1 -Dcl_khr_d3d11_sharing=1 -Dcl_khr_dx9_media_sharing=1 -Dcl_khr_image2d_from_buffer=1 -Dcl_khr_subgroups=1 -Dcl_khr_gl_event=1 -Dcl_khr_depth_images=1 -Dcl_khr_mipmap_image=1 -Dcl_khr_mipmap_image_writes=1 -Dcl_amd_liquid_flash=1 -Dcl_amd_copy_buffer_p2p=1 -Dcl_amd_planar_yuv=1 
*/


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

    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) {
            int cc = bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc] = acc[i][j];
        }
    }
}


__constant int pyopencl_defeat_cache_dd25ef165ef74402bf007c739e8ad015 = 0;