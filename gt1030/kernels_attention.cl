#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// (B,S,C) row-major -> (B,H,S,D) with optional pre-scale; C = H*D.
__kernel void split_heads(__global const half* src, __global half* dst,
                          const int C, const int S, const int H, const int D,
                          const float scale)
{
    int i = get_global_id(0);
    int d = i % D;
    int s = (i / D) % S;
    int h = (i / (D * S)) % H;
    int b = i / (D * S * H);
    dst[((b * H + h) * S + s) * D + d] =
        (half)((float)src[(b * S + s) * C + h * D + d] * scale);
}

// (B,H,S,D) -> (B,S,C)
__kernel void merge_heads(__global const half* src, __global half* dst,
                          const int C, const int S, const int H, const int D)
{
    int i = get_global_id(0);
    int d = i % D;
    int s = (i / D) % S;
    int h = (i / (D * S)) % H;
    int b = i / (D * S * H);
    dst[(b * S + s) * C + h * D + d] =
        src[((b * H + h) * S + s) * D + d];
}

// row-wise softmax over cols, half in/out (fp32 accumulation)
__kernel void softmax_rows(__global const half* src, __global half* dst,
                           const int rows, const int cols)
{
    int r = get_group_id(0);
    int lid = get_local_id(0);
    int ls = get_local_size(0);
    if (r >= rows) return;
    __local float lmax[256];
    __local float lsum[256];
    __global const half* row = src + r * cols;
    float m = -1e30f;
    for (int c = lid; c < cols; c += ls) m = fmax(m, (float)row[c]);
    lmax[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = ls / 2; step > 0; step >>= 1) {
        if (lid < step) lmax[lid] = fmax(lmax[lid], lmax[lid + step]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    m = lmax[0];
    float s = 0.f;
    for (int c = lid; c < cols; c += ls) s += exp((float)row[c] - m);
    lsum[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = ls / 2; step > 0; step >>= 1) {
        if (lid < step) lsum[lid] += lsum[lid + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.f / lsum[0];
    __global half* out = dst + r * cols;
    for (int c = lid; c < cols; c += ls)
        out[c] = (half)(exp((float)row[c] - m) * inv);
}

// nearest-neighbour 2x upsample, NCHW fp16
__kernel void upsample_nearest2x(__global const half* src, __global half* dst,
                                 const int N, const int C, const int H, const int W)
{
    int i = get_global_id(0);
    int W2 = 2 * W, H2 = 2 * H;
    int total = N * C * H2 * W2;
    if (i >= total) return;
    int w = i % W2;
    int h = (i / W2) % H2;
    int c = (i / (W2 * H2)) % C;
    int n = i / (W2 * H2 * C);
    dst[i] = src[((n * C + c) * H + h / 2) * W + w / 2];
}
