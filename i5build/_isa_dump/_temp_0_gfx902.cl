/* Compiler options:
-save-temps -cl-fast-relaxed-math -I C:\Users\Gan\AppData\Local\Programs\Python\Python311\Lib\site-packages\pyopencl\cl -D__AMD__=1 -D__gfx902__=1 -D__gfx902=1 -DFP_FAST_FMAF=1 -DFP_FAST_FMA=1 -D__IMAGE_SUPPORT__=1 -cl-denorms-are-zero -m64 -Dcl_khr_fp64=1 -Dcl_amd_fp64=1 -Dcl_khr_global_int32_base_atomics=1 -Dcl_khr_global_int32_extended_atomics=1 -Dcl_khr_local_int32_base_atomics=1 -Dcl_khr_local_int32_extended_atomics=1 -Dcl_khr_int64_base_atomics=1 -Dcl_khr_int64_extended_atomics=1 -Dcl_khr_3d_image_writes=1 -Dcl_khr_byte_addressable_store=1 -Dcl_khr_fp16=1 -Dcl_khr_gl_sharing=1 -Dcl_khr_gl_depth_images=1 -Dcl_amd_device_attribute_query=1 -Dcl_amd_vec3=1 -Dcl_amd_printf=1 -Dcl_amd_media_ops=1 -Dcl_amd_media_ops2=1 -Dcl_amd_popcnt=1 -Dcl_khr_d3d10_sharing=1 -Dcl_khr_d3d11_sharing=1 -Dcl_khr_dx9_media_sharing=1 -Dcl_khr_image2d_from_buffer=1 -Dcl_khr_subgroups=1 -Dcl_khr_gl_event=1 -Dcl_khr_depth_images=1 -Dcl_khr_mipmap_image=1 -Dcl_khr_mipmap_image_writes=1 -Dcl_amd_liquid_flash=1 -Dcl_amd_copy_buffer_p2p=1 -Dcl_amd_planar_yuv=1 
*/


#define TS 64
#define TK 8
__kernel void probe(__global const float* A, __global float* C) {
    __local float As[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    As[tx % TK][ty * 8 + tx % 8] = A[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < TK; k++) acc = fma(As[k][ty * 8 + tx % 8], 1.0001f, acc);
    C[get_global_id(0)] = acc;
}


__constant int pyopencl_defeat_cache_045c41ce87344ef89af142cca95ea67b = 0;