# 1 "C:\\Users\\Gan\\AppData\\Local\\Temp\\\\OCL3100T0.cl"
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
# 1 "C:\\Users\\Gan\\AppData\\Local\\Temp\\\\OCL3100T0.cl" 2

#define TS 64
#define TK 8
__kernel void probe(__global const float* A, __global float* C) {
    __local float As[8][64 + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    As[tx % 8][ty * 8 + tx % 8] = A[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.f;
#pragma unroll
 for (int k = 0; k < 8; k++) acc = fma(As[k][ty * 8 + tx % 8], 1.0001f, acc);
    C[get_global_id(0)] = acc;
}


__constant int pyopencl_defeat_cache_045c41ce87344ef89af142cca95ea67b = 0;
