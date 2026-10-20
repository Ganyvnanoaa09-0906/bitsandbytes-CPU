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
    int r = get_global_id(0);
    if (r >= rows) return;
    __global const half* row = src + r * cols;
    float m = -1e30f;
    for (int c = 0; c < cols; ++c) m = fmax(m, (float)row[c]);
    float s = 0.f;
    for (int c = 0; c < cols; ++c) s += exp((float)row[c] - m);
    float inv = 1.f / s;
    __global half* out = dst + r * cols;
    for (int c = 0; c < cols; ++c) out[c] = (half)(exp((float)row[c] - m) * inv);
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
