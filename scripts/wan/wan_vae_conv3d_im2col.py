"""Wan VAE top Conv3d: iGPU im2col3d + Sgemm fp32 vs torch CPU conv3d."""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import ctypes
import sys
import time

import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.hybrid_gpu as G

SRC = r'''
__kernel void im2col3d_f32(__global const float* src, __global float* dst,
                           int N,int C,int T,int H,int W,int K,int Ho,int Wo,int To) {
    int idx=get_global_id(0);
    int rows=N*To*Ho*Wo; int cols=C*K*K*K;
    if(idx>=rows*cols)return;
    int r=idx/cols, c=idx%cols;
    int n=r/(To*Ho*Wo); int rem=r%(To*Ho*Wo);
    int ot=rem/(Ho*Wo); int ohw=rem%(Ho*Wo);
    int oh=ohw/Wo, ow=ohw%Wo;
    int cch=c/(K*K*K); int kk=c%(K*K*K);
    int kt=kk/(K*K), kh=(kk/K)%K, kw=kk%K;
    int it=ot+kt, ih=oh+kh, iw=ow+kw;
    float v=0.f;
    if(it<T && ih<H && iw<W) v=src[(((n*C+cch)*T+it)*H+ih)*W+iw];
    dst[idx]=v;
}
'''


def bench(N,C,T,H,W,Cout,K=3):
    G.init(); ctx,q=G._ctx,G._q; mf=cl.mem_flags
    prog=cl.Program(ctx,SRC).build(options=['-cl-mad-enable']); kern=cl.Kernel(prog,'im2col3d_f32')
    bl=G._bl; qp=ctypes.c_void_p(int(q.int_ptr))
    To=T-K+1; Ho=H-K+1; Wo=W-K+1
    x=np.random.randn(N,C,T,H,W).astype(np.float32)
    w=np.random.randn(Cout,C,K,K,K).astype(np.float32)
    wmat=np.ascontiguousarray(w.reshape(Cout,C*K*K*K).T)
    rows=N*To*Ho*Wo; cols=C*K*K*K
    xb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=x)
    wb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=wmat)
    cb=cl.Buffer(ctx,mf.WRITE_ONLY,rows*cols*4)
    ob=cl.Buffer(ctx,mf.WRITE_ONLY,rows*Cout*4)
    out=np.empty((rows,Cout),np.float32)
    def run():
        kern(q,(rows*cols,),(256,),xb,cb,np.int32(N),np.int32(C),np.int32(T),np.int32(H),np.int32(W),np.int32(K),np.int32(Ho),np.int32(Wo),np.int32(To))
        rc=bl.CLBlastSgemm(101,111,111,rows,Cout,cols,1.0,ctypes.c_void_p(int(cb.int_ptr)),0,cols,ctypes.c_void_p(int(wb.int_ptr)),0,Cout,0.0,ctypes.c_void_p(int(ob.int_ptr)),0,Cout,ctypes.byref(qp),None)
        if rc: raise RuntimeError(rc)
    run(); q.finish(); cl.enqueue_copy(q,out,ob).wait()
    xt=torch.from_numpy(x); wt=torch.from_numpy(w)
    for _ in range(2): ref=F.conv3d(xt,wt)
    ts=[]
    for _ in range(3):
        t=time.perf_counter(); ref=F.conv3d(xt,wt); ts.append(time.perf_counter()-t)
    cpu=min(ts)*1e3
    ts=[]
    for _ in range(3):
        t=time.perf_counter(); run(); q.finish(); ts.append(time.perf_counter()-t)
    gpu=min(ts)*1e3
    got=out.reshape(N,To,Ho,Wo,Cout).transpose(0,4,1,2,3)
    err=float(np.max(np.abs(got-ref.numpy())))
    gf=2*N*Cout*To*Ho*Wo*C*K*K*K/1e9
    print('N%d C%d T%d %dx%d out%d: CPU %.2fms iGPU %.2fms speedup %.2fx GF %.2f err %.2e'%(N,C,T,H,W,Cout,cpu,gpu,cpu/gpu,gf,err),flush=True)

if __name__=='__main__':
    bench(1,192,4,32,32,192)
    bench(1,96,4,64,64,96)
