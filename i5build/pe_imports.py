"""List the imported DLLs of the built CPU library, from the PE import table.

The question is narrow and load-bearing: cpu_ops.cpp guards its `omp parallel for`
pragmas behind `#if defined(HAS_OPENMP) || defined(_OPENMP)`, so if this DLL was built
without OpenMP every one of those loops is serial. A string scan for "*.dll" is not
enough to answer it -- MSVC stores import names in mixed case and a statically linked
runtime leaves no trace at all -- so this parses the import directory properly and
also reports whether the OpenMP entry points are referenced anywhere in the binary.
"""
import struct
import sys


def rva_to_off(pe, rva, sections):
    for va, vsize, raw, rawsize in sections:
        if va <= rva < va + max(vsize, rawsize):
            return raw + (rva - va)
    return None


def parse(path):
    with open(path, "rb") as fh:
        d = fh.read()
    if d[:2] != b"MZ":
        raise SystemExit("not a PE file")
    e_lfanew = struct.unpack_from("<I", d, 0x3C)[0]
    assert d[e_lfanew:e_lfanew + 4] == b"PE\0\0", "bad PE signature"
    coff = e_lfanew + 4
    nsec, = struct.unpack_from("<H", d, coff + 2)
    opt_size, = struct.unpack_from("<H", d, coff + 16)
    opt = coff + 20
    magic, = struct.unpack_from("<H", d, opt)
    pe32p = magic == 0x20B
    # data directory 1 = imports
    dd = opt + (112 if pe32p else 96)
    imp_rva, imp_size = struct.unpack_from("<II", d, dd + 8)
    sec = opt + opt_size
    sections = []
    for i in range(nsec):
        off = sec + i * 40
        name = d[off:off + 8].rstrip(b"\0").decode("latin1")
        vsize, va, rawsize, raw = struct.unpack_from("<IIII", d, off + 8)
        sections.append((va, vsize, raw, rawsize))

    out = []
    off = rva_to_off(d, imp_rva, sections)
    if off is None:
        return out
    while True:
        desc = d[off:off + 20]
        if len(desc) < 20 or desc == b"\0" * 20:
            break
        name_rva, = struct.unpack_from("<I", d, off + 12)
        no = rva_to_off(d, name_rva, sections)
        if no is None:
            break
        end = d.index(b"\0", no)
        out.append(d[no:end].decode("latin1"))
        off += 20
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"=== {path} ===")
        try:
            libs = parse(path)
        except SystemExit as exc:
            print("  ", exc)
            continue
        for l in libs:
            print("   ", l)
        low = [l.lower() for l in libs]
        omp = [l for l in low if "omp" in l or "vcomp" in l]
        print(f"    OpenMP runtime: {omp if omp else 'NONE'}")
        with open(path, "rb") as fh:
            blob = fh.read()
        for sym in (b"omp_get_max_threads", b"omp_get_thread_num", b"__kmpc_fork_call",
                    b"GOMP_parallel", b"vcomp_for_static_simple"):
            if sym in blob:
                print(f"    symbol present: {sym.decode()}")
