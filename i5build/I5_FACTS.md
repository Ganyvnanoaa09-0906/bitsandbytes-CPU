# i5-10400 (DESKTOP-2J5D1P7) -- established facts

Gathered 2026-09-27 over ssh. Every line below was measured on that machine,
not inferred from the CPU model name.

## Hardware / OS

| Item | Value |
|---|---|
| CPU | Intel Core i5-10400 @ 2.90 GHz |
| Cores / threads | 6 / 12 |
| L2 / L3 | 1.5 MB / 12 MB |
| RAM | 11.79 GB total (7.41 GB free as measured) |
| OS | Windows 10 Pro 10.0.19045 (build 19045) |
| IP | 10.37.193.151 (WLAN) |
| User | GanYv |
| PowerShell | 5.1.19041.6456 (.NET Framework 4.0.30319) |
| Disk | C: 49.6 GB free / 111.1 GB; D: 293.9 GB free / 465.8 GB |

## SIMD: what can be built and what actually executes

Measured by compiling a small intrinsic-using program per level and RUNNING it.
The run exit code separates the two cases that matter:

- `0` -> this CPU executes that level
- `0xC000001D` (STATUS_ILLEGAL_INSTRUCTION) -> the binary does contain that
  level, this CPU cannot execute it
- build failure -> the compiler refused to emit it

| Level | Build | Run on this box |
|---|---|---|
| SSE2 | OK | OK |
| AVX2 (+FMA, F16C) | OK | OK |
| **AVX-512F** | OK | **CRASHES: 0xC000001D (illegal instruction)** |

### AVX-512: NO — and this corrects an earlier wrong claim

**The i5-10400 has no AVX-512.** CPUID read directly on the machine:

```
SSE2=1 AVX=1 FMA=1 F16C=1 OSXSAVE=1
AVX2=1 AVX512F=0 AVX512DQ=0 AVX512BW=0 AVX512VL=0 AVX512CD=0
XCR0=0x1f  YMM_enabled=1  ZMM_enabled=0
```

Neither the CPU advertises the feature bits nor has the OS enabled ZMM state.
That is expected for consumer Comet Lake; AVX-512 on that generation is limited
to the Xeon W-1200 parts.

The project's own notes previously asserted the opposite ("i5-10400 有
AVX512F/VL/BW/DQ/CD"). That is wrong and the planned "dual-TU + runtime
dispatch to use AVX-512 on the Intel box" work is **unnecessary** -- there is
nothing there to dispatch to.

**How the wrong answer nearly survived.** Three separate measurements disagreed,
and only the last one is authoritative:

| Measurement | Result | Why it misleads |
|---|---|---|
| clang, tiny probe with `-mavx512f` | "runs" | the optimizer folded it away; assembly had **no zmm** |
| clang, real-work probe | run_rc=0 | reported as if it proved support -- it did not; this was the false positive |
| MSVC `/arch:AVX512`, real work | **0xC000001D** | correct: assembly *did* contain zmm, so the crash is the answer |
| **CPUID feature bits** | **AVX512F=0, ZMM_enabled=0** | **authoritative** |

Two lessons worth keeping:
1. "It compiled and ran" is not evidence that the instruction was executed. The
   first probe proved nothing because the code was folded away; only the
   disassembly settled it.
2. When results contradict, go to the authoritative source (CPUID) instead of
   picking the measurement that fits the expectation. Two of the four
   measurements above pointed the wrong way.

### What the build should target here

`AVX2 + FMA + F16C` is the ceiling on this machine. Note that
`clang -march=native` picked `-target-cpu skylake`, which is a reasonable
baseline; and MSVC `/arch:AVX2` is the correct flag. Neither leaves anything on
the table now that AVX-512 is ruled out.

### The shell-quoting trap that corrupted these measurements

The CPUID probe above was first written as a `.cmd` that generated C source with
`echo` lines. The parentheses and `&` in `(r[1]>>16)&0xFF` were interpreted as
cmd syntax, so the source file was mangled, `cl` compiled garbage, and the run
produced an error that looked like a hardware limitation. That is the third time
in this session that shell quoting produced a wrong technical conclusion. The
fix: write source files from PowerShell (base64/here-string), never through cmd
`echo`, and keep every `.ps1` pure ASCII.


## Directory layout: the package is a SUBDIRECTORY of the checkout root

Read off a directory listing after three acceptance runs failed while guessing at
it. Every item below cost at least one full run, so they are written down:

```
C:\Users\GanYv\bnb_repo\                     <- checkout root (= the R5's
│                                                 D:\work\bitsandbytes-CPU)
├── csrc\cpu_ops.cpp, cpu_ops.h, cpu_gdn.cpp
├── build_manual\build_manual.bat            writes bitsandbytes\libbitsandbytes_cpu.dll
├── *.obj, bench_*.exe, selftest_cpu.c, selftest_i5.exe
├── _testpath.py                             sys.path bootstrap shared by the tests
├── train_1000_steps.py, run_all_tests.py, efst.py
├── disk_balancer.py, latent_chunk_store.py, temporal_only_lora.py
├── verify_*.py, stress_opt.py, test_temporal_only_lora.py
└── bitsandbytes\                            <- THE PACKAGE
    ├── __init__.py, cextension.py, gdn_cpu.py, fused_cpu.py
    ├── libbitsandbytes_cpu.dll              <- where the build lands
    └── autograd\ backends\ diagnostics\ nn\ optim\
```

The shape matches the R5's -- *a* checkout root containing a `bitsandbytes\`
package -- but the two roots are different directories, and that is precisely
what makes a hard-coded path fail:

| | R5 | i5 |
|---|---|---|
| checkout root | `D:\work\bitsandbytes-CPU` | `C:\Users\GanYv\bnb_repo` |
| package | `<root>\bitsandbytes` | `<root>\bitsandbytes` |

Gotchas, each observed on a real run:

1. **`import bitsandbytes` needs the checkout root on `sys.path`, NOT the package
   directory.** Putting the package directory there yields
   `ModuleNotFoundError: No module named 'bitsandbytes'`, because the importable
   name lives *inside* it. This broke `stress_opt.py`, `verify_e2e_train.py` and
   five of eight checks in `verify_bnb_intact.py` on one run, while the DLL and
   every kernel were perfectly fine.

2. **The R5 additionally has a `.pth` file** in its site-packages adding
   `D:\work\bitsandbytes-CPU\bitsandbytes` to `sys.path`, so on the R5 the
   package directory happens to be importable on its own. Nothing equivalent
   exists on the i5, so tests must not depend on it. `_testpath.py` sidesteps the
   question entirely by searching for a parent that works.

3. **`build_manual.bat` writes to `bitsandbytes\libbitsandbytes_cpu.dll`
   relative to its working directory** (the checkout root), and falls back to
   that path when the location named in its own comment does not exist. So the
   DLL lands next to `__init__.py`, which is where the ctypes loader looks. Its
   comment header claims `bitsandbytes\bitsandbytes\...`: the comment is stale,
   not the code.

4. `run_all_tests.py` locates child scripts **relative to its own directory**, so
   the test scripts must sit next to it, in the checkout root.

## Encoding: the failure path was the one path that could not run

On this machine a redirected console is codepage 936 (GBK), and the test scripts
print U+2705/U+274C check marks. GBK cannot encode those, so:

- `verify_bnb_intact.py` raised `UnicodeEncodeError` **inside `check()`, on the
  failure branch** -- so a failing check crashed the script instead of being
  reported;
- nothing called `sys.exit`, so it exited **0** anyway and the runner recorded
  `OK 11/12` for a run that had verified almost nothing;
- `run_all_tests.py` then hit the same thing while printing a child's captured
  output, 2.2 s into the sweep.

Three fixes, all now in place: `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` set by
`accept_i5.cmd` for the whole chain, `sys.stdout.reconfigure(encoding='utf-8')`
in the scripts, and a plain-ASCII `[PASS]`/`[FAIL]` fallback if reconfigure is
unavailable. `run_all_tests.py` also forces the encoding into every child's
environment. The general lesson: **an exception raised while reporting a failure
converts a visible failure into an invisible one**, and a script with no exit
code is not a test.

## Toolchain

**No Windows SDK is installed.** `WindowsSdkDir` is empty, `malloc.h` does not
exist anywhere on C: or D:, and only the MSVC compiler headers are present. So:

| Tool | Path | Status |
|---|---|---|
| `cl.exe` | `D:\vs\VC\Tools\MSVC\{14.44,14.51,14.52}\bin\HostX64\x64\cl.exe` | present but **cannot compile** (`fatal error C1083: malloc.h`) |
| `clang.exe` | `D:\vs\VC\Tools\Llvm\x64\bin\clang.exe` (22.1.3) | present but **cannot compile** (`'stdlib.h' file not found`) -- it targets `x86_64-pc-windows-msvc` and wants the SDK |
| **llvm-mingw clang 23.1.0** | `D:\toolchains\llvm-mingw-20260826-ucrt-x86_64\bin\clang.exe` | **works** -- target `x86_64-w64-windows-gnu`, self-contained (mingw-w64 headers + UCRT), builds and runs hello-world with no SDK |
| Python | `D:\scoop\apps\python\current\python.exe` (3.14.7) | pip 26.2.1 present; **no packages installed** (no numpy/torch/psutil) |
| git | `D:\scoop\apps\git\current\cmd\git.exe` | present |
| cmake | -- | not installed |

`vcvars64.bat` exists and initialises correctly (`VC Tools 14.51.36231`), but it
cannot help because the SDK is missing.

## Network (from the i5)

| Target | Result |
|---|---|
| github.com, raw.githubusercontent.com, ghproxy.com, mirror.ghproxy.com, hub.fastgit | **timeout (~12 s)** |
| **gh-proxy.com** | **OK, ~0.8 s** |
| **ghfast.top** | **OK, ~1.1 s** |
| pypi.org | OK, ~0.5 s |
| nuget.org | OK, ~2 s |
| aka.ms, registry.npmjs.org | OK |
| BITS transfer | fails with `0x800704DD` ("no logged-on network user") -- use `Invoke-WebRequest` instead |

Practical consequence: GitHub-hosted assets must be fetched through
`https://gh-proxy.com/https://github.com/...`. scoop cannot install anything
whose payload lives on github releases; that is exactly why the mingw install
failed the first time.

## Access

Passwordless ssh from the dev box (R5) works:

```
ssh i5 "command"                 # alias configured in ~/.ssh/config on the R5
scp file i5:C:/path/             # verified both directions
```

Key: `~/.ssh/id_ed25519_i5` on the R5 (empty passphrase), public half installed
in `C:\ProgramData\ssh\administrators_authorized_keys` on the i5 with the ACL
restricted to SYSTEM + Administrators (sshd silently ignores that file if its
permissions are too broad).

## Open items

1. **MSVC now works.** VS BuildTools 17.14 with the VCTools workload installed to
   `D:\vs2022bt`, MSVC `14.44.35207`, Windows SDK `10.0.26100.0` (with
   `ucrt\malloc.h`). `vcvars64.bat` initialises, `cl` compiles and runs
   hello-world, and `WindowsSdkDir` resolves. Note the SDK landed in the default
   location (`C:\Program Files (x86)\Windows Kits\10`), not on D:.

   The i5's pre-existing `D:\vs` still lacks the SDK, so use
   `D:\vs2022bt\VC\Auxiliary\Build\vcvars64.bat` in build scripts.

2. **AVX-512 is not available on this CPU** (see above). The "Intel support"
   roadmap item therefore reduces to: make sure an AVX2 build is correct and
   complete on Intel, and confirm the runtime CPU detection behaves. No AVX-512
   dispatch work is warranted.

3. **Python has no packages** -- `pip install numpy psutil` is needed before any
   benchmarking script runs there.

4. **A second, self-contained toolchain is present** at
   `D:\toolchains\llvm-mingw-20260826-ucrt-x86_64` (clang 23.1.0, target
   `x86_64-w64-windows-gnu`). Useful because it needs no SDK, but its output is
   GNU-ABI, so it cannot produce the MSVC-ABI DLL that `build_manual.bat` makes.

