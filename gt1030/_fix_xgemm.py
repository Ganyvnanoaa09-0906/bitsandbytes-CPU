# -*- coding: utf-8 -*-
"""把 CLBlast Xgemm 覆盖参数接到 ar_v2_multi 的 Dev 里。

实测（M=2048，本机 gfx902）：默认 KREG=1 时 4 个前向 GEMM 只有 124~258 GFLOPS；
KREG=32 时 442~517 GFLOPS（3.5×），反向 tn/nt 形状 690 GFLOPS 以上。
配置由 research/bench_results/clblast_best_configs_1030.json 出发、按我们的真实形状重扫得到。
可用 AR_XGEMM_KREG 环境变量改（0 = 不覆盖）。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""KNAMES = ('rms_stats',""", '''XGEMM_PN = ["GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG",
             "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"]


def xgemm_override(kreg=None, bl=None, ctx=None):
    """按我们真实形状重扫出来的 Xgemm 参数（KREG=32 最好，见 ar_v2_gemm_sweep*.py）。"""
    kreg = int(os.environ.get('AR_XGEMM_KREG', '32')) if kreg is None else int(kreg)
    if kreg <= 0:
        return -1
    bl = bl or A.bl
    ctx = ctx or A.ctx
    bl.CLBlastOverrideParameters.restype = ctypes.c_int
    bl.CLBlastOverrideParameters.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
    vals_d = dict(GEMMK=0, KREG=kreg, KWG=16, KWI=2, MDIMA=16, MDIMC=16, MWG=128,
                  NDIMB=16, NDIMC=16, NWG=128, SA=1, SB=1, STRM=1, STRN=1, VWM=4, VWN=2)
    names = (ctypes.c_char_p * len(XGEMM_PN))(*[x.encode() for x in XGEMM_PN])
    vals = (ctypes.c_size_t * len(XGEMM_PN))(*[int(vals_d[k]) for k in XGEMM_PN])
    return bl.CLBlastOverrideParameters(ctypes.c_void_p(int(ctx.devices[0].int_ptr)),
                                        b'Xgemm', 16, len(XGEMM_PN), names, vals)


KNAMES = ('rms_stats',''')

# Dev.__init__ 里应用（含 AR_NOOVERRIDE 关闭）；用到的 ctypes 已 import
s = s.replace("""        self._off = ALIGN
        self._health()""",
"""        self._off = ALIGN
        if os.environ.get('AR_NOOVERRIDE', '0') != '1':
            rc = xgemm_override(bl=None, ctx=self.ctx)
            if rc == 0:
                print('[xgemm] override KREG=%s applied' % os.environ.get('AR_XGEMM_KREG', '32'),
                      flush=True)
        self._health()""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched xgemm override')
