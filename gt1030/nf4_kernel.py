"""NF4 dequant+transpose kernel for hybrid 8B training (blocksize=64, nf4).

Input: packed uint8 (out, in/2) row-major, absmax fp32 (out, in/64), nf4 code table.
Output: W^T fp16 (in, out) row-major, tiled 32x32 for coalesced writes.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pyopencl as cl

import gt1030.hybrid_gpu as G

SRC = r'''
__constant float NF4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};
__kernel __attribute__((reqd_work_group_size(32, 32, 1)))
void nf4_dequant_transpose(__global const uchar* restrict q,
                           __global const float* restrict absmax,
                           __global half* restrict wt,
                           const int out, const int in) {
    __local half tile[32][33];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int i0 = get_group_id(0) * 32, o0 = get_group_id(1) * 32;
    const int i = i0 + tx, o = o0 + ty;
    if (i < in && o < out) {
        uchar b = q[o * (in / 2) + (i >> 1)];
        int nib = (i & 1) ? (b & 0x0F) : (b >> 4);
        float a = absmax[o * (in / 64) + (i >> 6)];
        tile[ty][tx] = (half)(NF4[nib] * a);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int oo = o0 + tx, ii = i0 + ty;
    if (oo < out && ii < in) wt[ii * out + oo] = tile[tx][ty];
}
'''

_prog = None


def _prog_init():
    global _prog
    G.init()
    if _prog is None:
        _prog = cl.Program(G._ctx, SRC).build(options=['-cl-mad-enable'])
        _prog.nf4_dequant_transpose = cl.Kernel(_prog, 'nf4_dequant_transpose')


def dequant_transpose_dev(packed: np.ndarray, absmax: np.ndarray, out: int, in_: int):
    """returns a device buffer holding W^T (in, out) fp16."""
    _prog_init()
    qbuf = G._buf(packed.size, 10)
    abuf = G._buf(absmax.size * 4, 11)
    wbuf = G._buf(in_ * out * 2, 12)
    cl.enqueue_copy(G._q, qbuf, np.ascontiguousarray(packed.reshape(-1)))
    cl.enqueue_copy(G._q, abuf, np.ascontiguousarray(absmax.reshape(-1).astype(np.float32)))
    gws = (((in_ + 31) // 32) * 32, ((out + 31) // 32) * 32)
    _prog.nf4_dequant_transpose(G._q, gws, (32, 32), qbuf, abuf, wbuf, np.int32(out), np.int32(in_))
    return wbuf
