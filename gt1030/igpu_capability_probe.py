"""igpu_capability_probe.py — runtime capability probe for Schrodinger's iGPU.

Answers the questions that decide whether the iGPU can host a workload at all:
  * device / driver / OpenCL version
  * global memory, single-allocation limit, local memory, max work-group
  * fp16 support (cl_khr_fp16 + an actual compiled fp16 kernel)
  * whether a packed NF4 Qwen-Image transformer (3.4 GB) fits in one allocation
  * a small CLBlast Sgemm/Hgemm smoke test

Run locally; no cloud upload.
"""
from __future__ import annotations

import numpy as np
import pyopencl as cl

import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')


def main():
    print('=== OpenCL devices ===')
    for pi, platform in enumerate(cl.get_platforms()):
        for di, dev in enumerate(platform.get_devices()):
            name = dev.name.strip()
            print('[%d.%d] %s | %s | driver %s' % (pi, di, name, dev.vendor,
                                                   dev.driver_version))
            gmem = dev.global_mem_size
            max_alloc = dev.max_mem_alloc_size
            lmem = dev.local_mem_size
            print('     global %.2f GB | max alloc %.2f GB | local %.0f KB | '
                  'max WG %d' % (gmem / 1e9, max_alloc / 1e9, lmem / 1e3,
                                 dev.max_work_group_size))
            print('     fp16:', 'cl_khr_fp16' in dev.extensions,
                  '| subgroups:', 'cl_khr_subgroups' in dev.extensions,
                  '| images:', dev.image_support)
            print('     Qwen packed 3.42GB fits single alloc:',
                  max_alloc >= 3.42e9)
            print('     persistent absmax 0.45GB fits:',
                  max_alloc >= 0.45e9)

    print('\n=== fp16 kernel smoke test ===')
    try:
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)
        src = r'''
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
        __kernel void addh(__global const half* a, __global const half* b,
                           __global half* c, const int n)
        {
            int i = get_global_id(0);
            if (i < n) c[i] = a[i] + b[i];
        }
        '''
        prog = cl.Program(ctx, src).build()
        a = np.array([1, 2, 3, 4], dtype=np.float16)
        b = np.array([10, 20, 30, 40], dtype=np.float16)
        mf = cl.mem_flags
        ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        cb = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
        cl.Kernel(prog, 'addh')(queue, (4,), None, ab, bb, cb, np.int32(4))
        out = np.empty(4, dtype=np.float16)
        cl.enqueue_copy(queue, out, cb)
        queue.finish()
        print('fp16 add result:', out.tolist(), '| OK' if out.tolist() == [11, 22, 33, 44]
              else '| MISMATCH')
    except Exception as e:  # noqa: BLE001
        print('fp16 smoke test failed:', type(e).__name__, e)

    print('\n=== CLBlast smoke test ===')
    try:
        from opencl_fp16_ops import OpenCLFP16
        rt = OpenCLFP16.get()
        print('OpenCLFP16 device:', rt.device_name, '| max alloc %.2f GB'
              % (rt.max_alloc / 1e9))
        print('CLBlast Hgemm/Sgemm symbols:',
              hasattr(rt.bl, 'CLBlastHgemm'), hasattr(rt.bl, 'CLBlastSgemm'))
    except Exception as e:  # noqa: BLE001
        print('CLBlast runtime failed:', type(e).__name__, e)


if __name__ == '__main__':
    main()
