"""The `bitsandbytes-cpu` command: one entry point that answers "what is this and how do
I call it" without a browser.

Why a CLI rather than only a document: a reference that lives in a file is read by the
people who already know it exists. `bitsandbytes-cpu help` is discoverable from the
moment pip finishes.

WHY THIS MODULE IS NOT INSIDE THE PACKAGE
-----------------------------------------
`bitsandbytes/__init__.py` imports torch, and torch is deliberately not a dependency of
this distribution: on PyPI that name is the CUDA build, so depending on it would pull
2.5 GB of nvidia-* wheels onto a machine with no NVIDIA device. The direct consequence
is that a user can have this package installed with torch absent -- and that is exactly
the user who runs `help`, because the help is what says to install torch. Pointed at
`bitsandbytes.cli:main`, the console script died with
`ModuleNotFoundError: No module named 'torch'` before printing a word (observed on
Ubuntu, Python 3.14, after a plain `pip install` of the sdist).

So the reference lives in a module that imports nothing but the standard library.
`detect` and `doctor` load what they need out of the package directory by path, which
works with no torch; only `selftest` needs it, and says so instead of raising.

What it is built around is not the upstream API surface -- that is documented upstream --
but the things this fork adds and the way they have to be called on a CPU. Every example
under a heading is a call that was executed on an AVX2 machine before it was written
down, including the ones that are supposed to fail.

The output follows the shape people already know from tools like nmap: a usage line,
ALL-CAPS sections, one `thing: what it does` per line, no blank lines inside a section,
and the code for the entry indented under it. A signature alone does not tell anyone how
to use a feature; the example does.

Sections can be read one at a time (`bitsandbytes-cpu help 8bitopt`), and the same tables
render as markdown (`help --markdown`), so the terminal output and
docs_cpu/API_REFERENCE.md cannot drift apart.

    bitsandbytes-cpu help
    bitsandbytes-cpu help 8bitopt
    bitsandbytes-cpu detect
    bitsandbytes-cpu doctor
    bitsandbytes-cpu selftest
    bitsandbytes-cpu version
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
import textwrap

WIDTH = 78

USAGE = ("Usage: bitsandbytes-cpu {help|detect|doctor|selftest|version} [section]\n"
         "       bitsandbytes-cpu help 8bitopt      one section instead of all of them")

# Sections are (key, title, [(left, right, [example lines], lang), ...]).
#   key      the word to pass to `help <key>`
#   title    printed upper-case by help, as-is by --markdown
#   left     what you type or call
#   right    one clause about it
#   example  the actual call, executed by tools/verify_help_examples.py
#   lang     "python" (verified), "bash"/"text" (not executable)
# Keep `right` to a single clause; the examples carry the detail.
PY = "python"
SH = "bash"

SECTIONS = [
    ("intro", "What the fork adds", [
        ("CPU kernels",
         "the CUDA kernels are gone: AVX2 (x86-64) and NEON (ARM64) C, compiled into "
         "the wheel", [], PY),
        ("bitsandbytes.optim",
         "8-bit optimizers: ~2.3 bytes of state per parameter instead of 8; the 4-bit "
     "optimizer takes ~1.03", [], PY),
        ("bitsandbytes.nn",
         "4-bit and 8-bit layers that run on a cpu tensor", [], PY),
        ("bitsandbytes.functional",
         "the quantisation primitives plus the fused CPU kernels", [], PY),
        ("torch_cpu_kit",
         "env vars, thread counts, oneDNN, memory watch, DataLoader defaults", [], PY),
        ("detect_cpu",
         "one dict describing this machine and the settings that follow from it", [], PY),
        ("gdn_cpu",
         "fused Gated DeltaNet for Qwen3-Next / Qwen3.5 linear attention", [], PY),
        ("disk_balancer",
         "spills cold parameters to another disk when RAM runs out", [], PY),
        ("bitsandbytes-cpu",
         "this command; pip also installs it as bnb-cpu", [], PY),
    ]),
    ("start", "First run", [
        ("bitsandbytes-cpu detect",
         "what the CPU is, and the thread count it recommends", [], SH),
        ("bitsandbytes-cpu doctor",
         "which native library will load, and whether every kernel symbol is in it", [], SH),
        ("bitsandbytes-cpu selftest",
         "runs a 4-bit layer, an 8-bit optimizer step and the GDN kernel; exit 1 on failure",
         [], SH),
        ("bitsandbytes-cpu version", "package, python, torch, platform", [], SH),
        ("import bitsandbytes on a CPU host",
         "succeeds with no CUDA anywhere; the library loads on first kernel use", [
             "import bitsandbytes as bnb",
             "print(bnb.__version__, bnb.functional.has_avx512bf16())",
         ], PY),
    ]),
    ("4bit", "4-bit on CPU", [
        ("quantise one Linear layer",
         "assign Params4bit, then call .to('cpu') -- .to() is what quantises", [
             "import torch, torch.nn as nn, bitsandbytes as bnb",
             "",
             "src = nn.Linear(4096, 4096, bias=False)   # an fp32 layer to convert",
             "q = bnb.nn.Linear4bit(4096, 4096, bias=False,",
             "                      compute_dtype=torch.float32,   # see `help threads`",
             "                      quant_type='nf4', compress_statistics=True)",
             "q.weight = bnb.nn.Params4bit(src.weight.data, requires_grad=False,",
             "                             quant_type='nf4', compress_statistics=True)",
             "q = q.to('cpu')          # <- without this the first forward raises",
             "",
             "x = torch.randn(8, 4096) # your input",
             "y = q(x)                 # weight goes back to fp32 inside the forward",
         ], PY),
        ("what the forward does",
         "M=1 takes the fused gemv_4bit kernel; M>1 unpacks to fp32 and calls a dense GEMM",
         [], PY),
        ("quantise a whole model in place", "swap every nn.Linear, leave the head alone", [
             "def to_4bit(module, skip=('lm_head',)):",
             "    for name, child in module.named_children():",
             "        if name in skip:",
             "            continue",
             "        if isinstance(child, nn.Linear):",
             "            new = bnb.nn.Linear4bit(child.in_features, child.out_features,",
             "                                    bias=child.bias is not None,",
             "                                    compute_dtype=torch.float32,",
             "                                    quant_type='nf4', compress_statistics=True)",
             "            new.weight = bnb.nn.Params4bit(child.weight.data,",
             "                                           requires_grad=False,",
             "                                           quant_type='nf4',",
             "                                           compress_statistics=True)",
             "            setattr(module, name, new.to('cpu'))",
             "        else:",
             "            to_4bit(child, skip)",
             "    return module",
             "",
             "to_4bit(model)",
             "print(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 'MB')",
         ], PY),
        ("save and load a 4-bit layer",
         "the state dict carries the packed weight and the quantisation state; a "
         "fresh Linear4bit cannot load it back, so the parameter is rebuilt", [
             "src = nn.Linear(4096, 4096, bias=False)",
             "q = bnb.nn.Linear4bit(4096, 4096, bias=False, compute_dtype=torch.float32,",
             "                      quant_type='nf4', compress_statistics=True)",
             "q.weight = bnb.nn.Params4bit(src.weight.data, requires_grad=False,",
             "                             quant_type='nf4', compress_statistics=True)",
             "q = q.to('cpu')",
             "x = torch.randn(8, 4096)",
             "y0 = q(x)",
             "",
             "torch.save(q.state_dict(), 'layer4bit.pt')   # weight, weight.absmax,",
             "                                            # weight.quant_map, nested_*",
             "",
             "sd = torch.load('layer4bit.pt')",
             "state = bnb.functional.QuantState.from_dict(",
             "    {k[len('weight.') :]: v for k, v in sd.items() if k.startswith('weight.')},",
             "    device=torch.device('cpu'))",
             "fresh = bnb.nn.Linear4bit(4096, 4096, bias=False,",
             "                          compute_dtype=torch.float32, quant_type='nf4')",
             "pw = bnb.nn.Params4bit(sd['weight'], requires_grad=False,",
             "                       quant_type=state.quant_type, blocksize=state.blocksize,",
             "                       bnb_quantized=True, quant_storage=sd['weight'].dtype)",
             "pw.quant_state = state",
             "fresh.weight = pw",
             "assert (fresh(x) - y0).abs().max() < 1e-5",
         ], PY),
        ("Embedding4bit",
         "the same for a token table; there is no padding_idx argument", [
             "# table: a [vocab, dim] matrix, e.g. model.get_input_embeddings().weight",
             "emb = bnb.nn.Embedding4bit(table.shape[0], table.shape[1], quant_type='nf4')",
             "emb.weight = bnb.nn.Params4bit(table, requires_grad=False, quant_type='nf4')",
             "emb = emb.to('cpu')",
         ], PY),
    ]),
    ("qlora", "Training a frozen 4-bit base", [
        ("freeze the base, train adapters",
         "4-bit is a storage format: the base still computes in fp32", [
             "to_4bit(model)                 # the helper from `help 4bit`",
             "for p in model.parameters():",
             "    p.requires_grad_(False)",
             "",
             "adapters = nn.Linear(8, 8)     # wherever your LoRA layers come from",
             "opt = AdamW8bit(adapters.parameters(), lr=1e-4)",
             "loss = loss_fn(adapters(model(x)), y)",
             "loss.backward()",
             "opt.step(); opt.zero_grad()",
         ], PY),
        ("4-bit does not make the matmul cheaper",
         "the forward unpacks to fp32, so FLOPs are unchanged; only RAM and disk shrink",
         [], PY),
        ("gradients into a 4-bit layer",
         "requires_grad=True on a Params4bit is not supported here; keep the base frozen",
         [], PY),
    ]),
    ("8bitopt", "8-bit optimizer (start here)", [
        ("swap the optimizer",
         "same constructor arguments as the torch optimizer it replaces", [
             "import bitsandbytes as bnb",
             "",
             "opt = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5, weight_decay=0.01)",
             "opt.zero_grad()",
             "loss = loss_fn(model(x), y)",
             "loss.backward()",
             "opt.step()               # the loop is otherwise unchanged",
         ], PY),
        ("what you get",
         "18688 bytes of state per parameter here, against torch AdamW's 65540 (3.5x)", [
             "p = nn.Parameter(torch.randn(8192))",
             "opt = bnb.optim.AdamW8bit([p], lr=1e-3)",
             "p.grad = torch.randn_like(p)",
             "opt.step()",
             "nbytes = sum(v.numel() * v.element_size()",
             "             for v in opt.state[p].values() if torch.is_tensor(v))",
             "print(nbytes, 'bytes for', p.numel(), 'params =', nbytes / p.numel(), 'B/param')",
             "print(sorted(opt.state[p]))     # absmax1 absmax2 qmap1 qmap2 state1 state2 step",
         ], PY),
        ("every optimizer has the same three names",
         "<Name>8bit, <Name>32bit and Paged<Name>8bit", [
             "from bitsandbytes.optim import (Adam8bit, AdamW8bit, AdamW32bit, Lion8bit,",
             "                                SGD8bit, LAMB8bit, LARS8bit, RMSprop8bit,",
             "                                Adagrad8bit, AdEMAMix8bit, PagedAdamW8bit)",
             "opt = Lion8bit(model.parameters(), lr=1e-4, weight_decay=0.1)",
         ], PY),
        ("only the parameters you step",
         "a frozen base needs no optimizer state; passing it wastes the memory you saved", [
             "opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad],",
             "                          lr=1e-4)",
         ], PY),
        ("min_8bit_size=4096",
         "anything smaller keeps float32 state, so tiny tensors show no saving", [
             "small = [nn.Parameter(torch.randn(64))]",
             "opt = bnb.optim.AdamW8bit(small, lr=1e-3)          # default: fp32 state",
             "p = small[0]",
             "p.grad = torch.randn_like(p)",
             "opt.step()",
             "print({k: str(v.dtype) for k, v in opt.state[p].items() if torch.is_tensor(v)})",
             "# {'state1': 'torch.float32', 'state2': 'torch.float32'} -- no absmax, no gain",
             "",
             "opt = bnb.optim.AdamW8bit(small, lr=1e-3, min_8bit_size=0)   # force 8-bit",
             "opt.step()",
             "print({k: str(v.dtype) for k, v in opt.state[p].items() if torch.is_tensor(v)})",
             "# {'state1': 'torch.uint8', ..., 'absmax1': 'torch.float32', ...}",
         ], PY),
        ("state_dict", "saves and loads like any other torch optimizer", [
             "opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)",
             "torch.save(opt.state_dict(), 'opt.pt')",
             "opt.load_state_dict(torch.load('opt.pt', weights_only=False))",
         ], PY),
        ("the step is not slower",
         "the fused kernel drops the dequantise-then-update round trip: 0.74x an fp32 "
         "AdamW step, measured in one process", [], PY),
    ]),
    ("4bitopt", "4-bit optimizer: half the state of 8-bit", [
        ("swap the optimizer",
         "same shape of API as the 8-bit one; fp32 parameters only", [
             "from bitsandbytes.optim import AdamW4bit",
             "",
             "opt = AdamW4bit(model.parameters(), lr=1e-4, weight_decay=0.01)",
             "opt.zero_grad()",
             "loss = loss_fn(model(x), y)",
             "loss.backward()",
             "opt.step()               # the loop is otherwise unchanged",
         ], PY),
        ("what you get",
         "1.031 bytes of state per parameter, against 2.062 for the 8-bit optimizer "
         "and 8.0 for torch AdamW, all measured at 1M parameters", [
             "p = nn.Parameter(torch.randn(4096))",
             "opt = AdamW4bit([p], lr=1e-3)",
             "p.grad = torch.randn_like(p)",
             "opt.step()",
             "print(opt.state_bytes(), 'bytes for', p.numel(), 'params =',",
             "      opt.state_bytes() / p.numel(), 'B/param')",
             "# 4224 bytes for 4096 params = 1.031 B/param   (torch AdamW: 8.0)",
             "",
             "# on a real model, 30.49M parameters: 232.6 MB -> 29.5 MB, 7.9x less",
         ], PY),
        ("state layout",
         "two 4-bit codes per byte, one absmax per 256 elements", [
             "# state1 / state2 : ceil(n/2) uint8, high nibble = even index",
             "# absmax1/absmax2 : one fp32 per 256 elements",
             "# 16 symmetric levels: code 0 is -1, code 15 is +1, step 2/15",
             "# exact zero is therefore not representable; an all-zero block writes",
             "# code 8 (+1/15), whose bias is absmax/15 == 0 because absmax is 0",
         ], PY),
        ("does it train as well as fp32",
         "it tracks fp32 rather than matching it: quantised state cannot reproduce an "
         "fp32 trajectory, and that is not the criterion", [
             "# measured over 120 steps on the real 30.49M image-AR model:",
             "#   the two loss curves stay within 0.27% of each other",
             "# on a 200-step MLP run the 4-bit run ended at 0.0198 against fp32's",
             "#   0.0240 -- neither worse nor identical, simply a different path",
             "# the kernel's state codes are identical to an independent numpy",
             "# implementation over 50 steps, so the maths agrees exactly",
         ], PY),
        ("the step is faster, unlike 8-bit",
         "0.658 ms against 1.682 ms for the 8-bit kernel at 1M parameters (2.56x), "
         "because it moves half the bytes", [
             "# both are AVX2 kernels in the same library, so the comparison isolates",
             "# state width rather than implementation language",
             "# the scalar fallback is 7.99 ms -- the AVX2 path is 12x faster than it",
         ], PY),
        ("traps",
         "the failure modes seen while building this, each one silent at first", [
             "# 1. torch.optim.AdamW is NOT replaced: this is a separate class, and",
             "#    AdamW4bit requires fp32 params (the kernel's dtype 0).",
             "# 2. blocksize is compiled in at 256; passing another value raises.",
             "# 3. if the shared library does not export",
             "#    coptimizer_update_4bit_blockwise_cpu, import fails loudly rather",
             "#    than falling back -- three earlier versions of the search silently",
             "#    picked a DLL without the kernel and only failed at the first step.",
             "# 4. quantised state is not a drop-in for exact reproducibility runs:",
             "#    same seed, same data, different trajectory from fp32.",
         ], PY),
    ]),
    ("kernels", "Fused kernels (call them directly)", [
        ("fused_dequant_linear_8bit(...)",
         "out = A @ dequant8(wq).T with the weight never unpacked to fp32; the "
         "arguments are (A, wq, absmax, blocksize)", [
             "w = torch.randn(512, 512) * 0.02              # your fp32 weight",
             "A = torch.randn(4, 512)                       # your activations",
             "code = bnb.functional.create_linear_map()     # the 8-bit codebook",
             "wq, state = bnb.functional.quantize_blockwise(w, code=code, blocksize=64)",
             "out = bnb.functional.fused_dequant_linear_8bit(A, wq, state.absmax, 64)",
             "ref = A @ w.t()",
             "assert ((out - ref).norm() / ref.norm()).item() < 0.02",
         ], PY),
        ("A may have leading dimensions",
         "A.shape == (..., K) in, (..., N) out -- flatten (B,T,H,W) yourself first", [
             "w = torch.randn(512, 512) * 0.02",
             "code = bnb.functional.create_linear_map()",
             "wq, state = bnb.functional.quantize_blockwise(w, code=code, blocksize=64)",
             "v = torch.randn(2, 3, 512)",
             "out = bnb.functional.fused_dequant_linear_8bit(v, wq, state.absmax, 64)",
             "print(out.shape)        # (2, 3, 512) -> (2, 3, 512)",
         ], PY),
        ("K % blocksize must be 0",
         "otherwise the kernel refuses rather than reading the wrong row", [], PY),
        ("optimizer_update_8bit_blockwise(...)",
         "the fused 8-bit step the optimizers call, with the whole argument list", [
             "N, BLOCK = 4096, 64",
             "p = nn.Parameter(torch.randn(N) * 0.1)",
             "g = torch.randn(N) * 0.05",
             "state1 = torch.zeros(N, dtype=torch.uint8)      # quantised exp_avg",
             "state2 = torch.zeros(N, dtype=torch.uint8)      # quantised exp_avg_sq",
             "absmax1 = torch.zeros(N // BLOCK)",
             "absmax2 = torch.zeros(N // BLOCK)",
             "qmap1 = bnb.functional.create_dynamic_map(signed=True, total_bits=8)",
             "qmap2 = qmap1.clone()",
             "bnb.functional.optimizer_update_8bit_blockwise(",
             "    'adam', g, p, state1, state2,",
             "    0.9, 0.999, 0.0,       # beta1, beta2, beta3",
             "    1.0, 1e-8,             # alpha, eps",
             "    1, 1e-3,               # step, lr",
             "    qmap1, qmap2, absmax1, absmax2,",
             "    weight_decay=0.0, gnorm_scale=1.0)",
             "# checked against torch.optim.AdamW in tests/test_cpu_e2e.py: the two agree",
             "# to 1e-5 after one step",
         ], PY),
        ("gemv_4bit(A, q, state=state)",
         "the M=1 4-bit path on its own; state comes from quantize_4bit", [
             "w = torch.randn(512, 512) * 0.02",
             "A = torch.randn(1, 512)",
             "q, state = bnb.functional.quantize_4bit(w, quant_type='nf4', blocksize=64,",
             "                                        compress_statistics=True)",
             "row = bnb.functional.gemv_4bit(A, q, state=state)",
             "print(q.shape, q.dtype, row.shape)      # packed uint8, one row out",
         ], PY),
        ("int8_linear_matmul(A, B)",
         "int8 x int8 -> int32, both operands (M,K) and (N,K); the LLM.int8() core", [
             "Ai = torch.randint(-100, 100, (8, 64), dtype=torch.int8)",
             "Bi = torch.randint(-100, 100, (16, 64), dtype=torch.int8)",
             "out = bnb.functional.int8_linear_matmul(Ai, Bi)",
             "assert torch.equal(out, Ai.to(torch.int32) @ Bi.to(torch.int32).t())",
         ], PY),
        ("int8_vectorwise_quant(A)",
         "returns (int8 data, per-row fp32 scales, outlier column indices or None)", [
             "A = torch.randn(256, 512)",
             "qi, scales, outliers = bnb.functional.int8_vectorwise_quant(A)",
             "print(qi.dtype, qi.shape, scales.shape)   # int8, (256, 512), (256,)",
         ], PY),
    ]),
    ("toolkit", "torch_cpu_kit and detect_cpu", [
        ("apply_env() BEFORE import torch",
         "OpenMP reads these once, when the runtime loads; later calls do nothing", [
             "import bitsandbytes.torch_cpu_kit as tck",
             "tck.apply_env()          # OMP/MKL_NUM_THREADS = physical cores, OMP_WAIT_POLICY",
             "import torch             # torch now starts with those settings",
         ], PY),
        ("init(model_kind=...) after import torch",
         "thread pools, interop=1, oneDNN on, fp32 matmul precision left alone on AVX2", [
             "import torch, bitsandbytes.torch_cpu_kit as tck",
             "print(tck.init(model_kind='lm'))          # 'lm', 'vision', 'diffusion', or None",
         ], PY),
        ("physical_cores()",
         "half of os.cpu_count() on an SMT machine -- the number to use for GEMM", [
             "torch.set_num_threads(tck.physical_cores())   # 6, not 12, on an i5-10400",
         ], PY),
        ("mem_report()", "(process RSS MB, system available MB)", [
             "rss, avail = tck.mem_report()",
             "print(f'{rss:.0f} MB in use, {avail:.0f} MB available')",
         ], PY),
        ("start_mem_monitor(interval=20.0)",
         "background log line; returns the Event that stops it", [
             "stop = tck.start_mem_monitor(interval=20.0)",
             "tck.suspend_mem_monitor(stop)      # or stop.set()",
         ], PY),
        ("fast_loader(dataset, batch_size=32)",
         "DataLoader with pin_memory=False and 2 persistent workers", [
             "loader = tck.fast_loader(dataset, batch_size=32)",
             "print(loader.num_workers, loader.pin_memory)      # 2 False",
         ], PY),
        ("make_contiguous_(model)",
         "in place: transposed parameters make the C kernels fall back to a slow path", [
             "tck.make_contiguous_(model)",
         ], PY),
        ("detect()", "the dict behind `bitsandbytes-cpu detect`", [
             "from bitsandbytes.detect_cpu import detect",
             "d = detect()",
             "threads = d['recommend_threads']['lm']       # also 'diffusion', 'image'",
             "print(d['cpu_capability'], d['physical_cores'], threads)",
         ], PY),
    ]),
    ("threads", "Threads and precision", [
        ("threads = physical cores",
         "measured on an i5-10400: 6 threads, 203 s per 1024x1024 image step; 12 threads, 225 s",
         [], PY),
        ("bf16 is emulated on AVX2",
         "no hardware bf16 means software conversion: 5-12x slower at 512x512, worse at LLM shapes",
         [], PY),
        ("compute_dtype=torch.float32",
         "for Linear4bit, Linear8bitLt and Embedding4bit; bf16 there is the single "
         "biggest performance mistake on these CPUs", [], PY),
        ("tck.autocast()",
         "a no-op on AVX2: recommended_autocast() returns float32, so the block stays fp32", [
             "with tck.autocast():            # nullcontext on this machine, real on AMX/AVX512-BF16",
             "    out = model(x)",
         ], PY),
        ("float32_matmul_precision",
         "left at 'highest' on AVX2; only a CPU with bf16 hardware is put on 'high'", [], PY),
    ]),
    ("memory", "Memory", [
        ("plan for the peak, not the weights",
         "a 4-bit model still materialises fp32 activations and one unpacked layer at a time",
         [], PY),
        ("start_mem_monitor", "a log line every 20 s, on a daemon thread", [], PY),
        ("RAM below ~8 GB free",
         "detect() sets recommend_disk_balancer, and `help disks` is the answer", [], PY),
        ("MIN_8BIT_SIZE",
         "the optimizer, not you, decides: below 4096 elements a parameter stays 32-bit", [], PY),
    ]),
    ("gdn", "Gated DeltaNet kernel (Qwen3-Next)", [
        ("patch_transformers()",
         "routes HuggingFace Qwen3-Next / Qwen3.5 through the fused kernel; True if applied", [
             "from bitsandbytes import gdn_cpu",
             "print(gdn_cpu.patch_transformers())   # True once the patch is in place",
             "",
             "# then load the model as usual -- the layer now runs on the fused kernel:",
             "# model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-Next-80B-A3B-Instruct')",
         ], PY),
        ("patch_fla()", "the same for the fla library", [], PY),
        ("load_native()",
         "loads the shared library without patching anything; raises with the paths it tried", [
             "from bitsandbytes import gdn_cpu",
             "lib = gdn_cpu.load_native()",
         ], PY),
        ("fused_recurrent_gated_delta_rule(...)",
         "the kernel itself; the arguments are q, k, v, beta, g, scale, initial_state, "
         "output_final_state, use_qk_l2norm_in_kernel, head_first, cu_seqlens", [], PY),
        ("it has an autograd Function",
         "so it trains, not just infers; the reference path is kept for verification", [], PY),
    ]),
    ("disks", "Disk balancer", [
        ("add_flash_args(parser)",
         "adds --flash, --flash-paths, --flash-sizes, --flash-speed, --flash-threshold "
         "and --flash-keep to your own training script", [], PY),
        ("parse_flash_args(args)",
         "args -> DiskBalancerConfig, or None when --flash was not given", [
             "import argparse, bitsandbytes.disk_balancer as db",
             "ap = argparse.ArgumentParser()",
             "db.add_flash_args(ap)",
             "cfg = db.parse_flash_args(ap.parse_args(['--flash']))",
             "print(cfg.mode, cfg.memory_threshold, cfg.min_param_size)",
         ], PY),
        ("DiskLoadBalancer(config)",
         "attach_model() then start(); update_step() once per training step checks RAM "
         "and migrates the cold parameters", [], PY),
        ("in a training loop",
         "the whole integration is three lines and one call per step", [
             "# cfg comes from parse_flash_args(...); see the entry above",
             "if cfg:",
             "    bal = db.DiskLoadBalancer(cfg).attach_model(model).start()",
             "    for xb, yb in loader:",
             "        loss = loss_fn(model(xb), yb)",
             "        loss.backward(); opt.step(); opt.zero_grad()",
             "        bal.update_step()",
         ], PY),
        ("cold tensors by hand",
         "put_cold(key, tensor) / get_cold(key) / drop(key) / stats(); put_cold frees "
         "the tensor by replacing its storage with an empty one", [
             "if cfg:",
             "    bal.put_cold('text_embeds', embeds)      # any tensor you want off the heap",
             "    embeds = bal.get_cold('text_embeds')",
             "    print(bal.stats())",
         ], PY),
        ("it is a context manager",
         "cleanup() flushes the write queue and removes the staging files", [
             "if cfg:",
             "    with db.DiskLoadBalancer(cfg) as bal:",
             "        ...",
         ], PY),
    ]),
    ("layers", "Reference: bitsandbytes.nn", [
        ("Linear4bit(input_features, output_features, bias=True, compute_dtype=None,\n"
         "           compress_statistics=True, quant_type='fp4',\n"
         "           quant_storage=torch.uint8, device=None)",
         "4-bit weights; quant_type='nf4' for model weights -- note that quant_type comes "
         "after compress_statistics, so pass it by name", [], PY),
        ("Linear8bitLt(input_features, output_features, bias=True, has_fp16_weights=True,\n"
         "             threshold=0.0, index=None, device=None)",
         "LLM.int8(): 8-bit matmul, outlier columns kept in fp16; needs has_fp16_weights=False",
         [], PY),
        ("LinearNF4(...) / LinearFP4(...)", "Linear4bit with quant_type already fixed", [], PY),
        ("Params4bit(data, requires_grad=False, quant_state=None, blocksize=None,\n"
         "           compress_statistics=True, quant_type='fp4',\n"
         "           quant_storage=torch.uint8, bnb_quantized=False)",
         "the weight type; it only quantises when .to(device) is called on it, and "
         "bnb_quantized=True tells it a checkpoint is already packed", [], PY),
        ("Int8Params(data, requires_grad=True, has_fp16_weights=False)",
         "the weight type of Linear8bitLt", [], PY),
        ("Embedding4bit(num_embeddings, embedding_dim, dtype=None, quant_type='fp4',\n"
         "              quant_storage=torch.uint8, device=None)",
         "quantised token table; no padding_idx, unlike nn.Embedding", [], PY),
        ("Embedding8bit(num_embeddings, embedding_dim, device=None, dtype=None)",
         "LLM.int8() token table", [], PY),
        ("OutlierAwareLinear(input_features, output_features, bias=True, device=None)",
         "Linear8bitLt's building block", [], PY),
    ]),
    ("optim", "Reference: bitsandbytes.optim", [
        ("families",
         "Adam AdamW AdEMAMix Adagrad LAMB LARS Lion RMSprop SGD", [], PY),
        ("variants", "<Name>8bit, <Name>32bit, Paged<Name>8bit, and the bare name", [], PY),
        ("AdamW8bit(params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01,\n"
         "           min_8bit_size=4096)",
         "the common case; identical signature to torch.optim.AdamW plus min_8bit_size",
         [], PY),
        ("8bit vs 32bit",
         "8bit quantises the optimizer state; 32bit is the same optimizer with fp32 state",
         [], PY),
        ("optim_bits=32 in that signature is not the state width",
         "with args=None the class name decides: AdamW8bit stores uint8 state1, AdamW32bit "
         "float32", [
             "p = nn.Parameter(torch.randn(8192))",
             "p.grad = torch.randn_like(p)",
             "for cls in (bnb.optim.AdamW8bit, bnb.optim.AdamW32bit):",
             "    o = cls([p], lr=1e-3)",
             "    o.step()",
             "    print(cls.__name__, {k: str(v.dtype) for k, v in o.state[p].items()",
             "                         if torch.is_tensor(v) and k.startswith('state')})",
             "# AdamW8bit  {'state1': 'torch.uint8',   'state2': 'torch.uint8'}",
             "# AdamW32bit {'state1': 'torch.float32', 'state2': 'torch.float32'}",
         ], PY),
        ("Paged<Name>8bit",
         "upstream's paged variant; on a CPU there is no unified memory to page to", [], PY),
        ("state_dict / load_state_dict", "supported; the 8-bit state survives a round trip", [], PY),
    ]),
    ("functional", "Reference: bitsandbytes.functional", [
        ("quantize_blockwise(A, code=None, blocksize=4096)",
         "-> (uint8 codes, QuantState); code=create_linear_map() for a linear 8-bit map", [], PY),
        ("dequantize_blockwise(A, quant_state)", "the inverse", [], PY),
        ("quantize_4bit(A, absmax=None, out=None, blocksize=None,\n"
         "              compress_statistics=False, quant_type='fp4',\n"
         "              quant_storage=torch.uint8)",
         "-> (packed uint8, QuantState); quantize_nf4 / quantize_fp4 fix the type", [], PY),
        ("dequantize_4bit(A, quant_state)",
         "the inverse; this is what the M>1 forward does", [], PY),
        ("fused_dequant_linear_8bit(A, wq, absmax, blocksize)",
         "FORK: dequantise and multiply in one AVX2 pass, weight stays uint8", [], PY),
        ("optimizer_update_8bit_blockwise(...)",
         "FORK: the CPU port of the fused 8-bit optimizer step", [], PY),
        ("gemv_4bit / igemm / batched_igemm / int8_linear_matmul",
         "the int8 and 4-bit matmul primitives", [], PY),
        ("int8_vectorwise_quant / int8_vectorwise_dequant / int8_double_quant",
         "the row-wise statistics LLM.int8() is built from", [], PY),
        ("create_linear_map / create_dynamic_map / create_normal_map / create_fp8_map",
         "build a quantisation codebook", [], PY),
        ("has_avx512bf16()",
         "whether this build has the AVX512-BF16 kernels; False on AVX2-only machines", [], PY),
        ("QuantState.from_dict(qs_dict, device)",
         "rebuild the quantisation state when loading a checkpoint by hand", [], PY),
    ]),
    ("notes", "Traps, and the errors they produce", [
        ("forgot .to('cpu')", "the layer exists but has no quantisation state yet", [
            "AssertionError: FP4 quantization state not initialized. Please call .cuda()",
            "or .to(device) on the LinearFP4 layer first.",
        ], "text"),
        ("K is not a multiple of blocksize",
         "the fused kernel refuses rather than reading the wrong row", [
            "ValueError: fused_dequant_linear_8bit requires K % blocksize == 0, got",
            "K=500 blocksize=64 (per-row block layout). Pad K or pick a blocksize that",
            "divides K.",
        ], "text"),
        ("quantised state dict, fresh layer",
         "rebuild the parameter instead: Params4bit(..., bnb_quantized=True), as in 4bit",
         [
            "RuntimeError: Unexpected key(s) in state_dict: weight.absmax,",
            "weight.quant_map, weight.nested_absmax, ...",
         ], "text"),
        ("compute_dtype=bfloat16 on AVX2",
         "5-12x slower at 512x512 and worse at LLM shapes; the weights stay 4-bit, so "
         "nothing is saved", [], PY),
        ("torch is not a dependency",
         "on PyPI that name is the CUDA build; install the CPU wheel first", [], PY),
        ("Linux compiles on install",
         "needs g++ and libomp; the artefact is built against your own glibc", [
             "sudo apt install g++ libomp-dev        # Debian / Ubuntu",
             "sudo dnf install gcc-c++ libomp-devel  # Fedora / RHEL",
         ], SH),
        ("num_workers > 0 on Windows",
         "spawns processes, so the entry script needs if __name__ == '__main__':", [
             "RuntimeError: An attempt has been made to start a new process before the",
             "current process has finished its bootstrapping phase.",
         ], "text"),
        ("unsupported",
         "no CUDA, no ROCm, no training through a 4-bit base on a CPU, no bf16 hardware "
         "on AVX2", [], PY),
    ]),
]

EXAMPLES = [
    "bitsandbytes-cpu detect",
    "bitsandbytes-cpu selftest",
    "bitsandbytes-cpu help 8bitopt",
    "bitsandbytes-cpu help --markdown > API_REFERENCE.md",
]


def _row(row):
    left, right = row[0], row[1]
    example = row[2] if len(row) > 2 else []
    lang = row[3] if len(row) > 3 else PY
    return left, right, list(example), lang


def _select(pattern):
    """(sections, error) for an optional `help <pattern>` argument."""
    if not pattern:
        return SECTIONS, None
    p = pattern.strip().lower()
    exact = [s for s in SECTIONS if s[0] == p]
    if exact:
        return exact, None
    hits = [s for s in SECTIONS if p in s[1].lower()]
    if hits:
        return hits, None
    return [], p


def _wrap(left, right, indent=2, width=WIDTH):
    """One entry, nmap-style.

    Two layouts, chosen by how long the left side is:
      short left  ->  `  left: right`, continuation aligned under `right`
      long left   ->  `  left:` then the description on its own lines at indent 6

    Aligning the continuation under the description is what nmap does and it reads well
    while the left side stays under about 30 columns. Doing it unconditionally is what
    the first version got wrong: a 70-column signature pushed the text to column 75 and
    every wrapped line became three words.
    """
    left_lines = left.split("\n")
    lines = [" " * indent + left_lines[0]]
    lines += [" " * (indent + 1) + l for l in left_lines[1:]]
    last = lines[-1]
    left_width = max(len(l) for l in lines) - indent

    # Inline whenever it fits, whatever the left width: refusing to use a line that has
    # room is how the previous version pushed "LinearNF4(...) / LinearFP4(...)" onto two
    # lines while a 75-column line sat there half empty.
    if len(last) + 2 + len(right) <= width:
        return lines[:-1] + [f"{last}: {right}"]

    # Otherwise the description needs its own lines. Align them under the description
    # only while that leaves a usable column; past about 40 the alignment eats the line.
    if left_width <= 40:
        avail = max(20, width - len(last) - 2)
        wrapped = textwrap.wrap(right, avail, break_on_hyphens=False)
        hang = " " * (len(last) + 2)
        return lines[:-1] + [f"{last}: {wrapped[0]}"] + [hang + w for w in wrapped[1:]]

    block = " " * 6
    lines[-1] = last + ":"
    return lines + [block + w
                    for w in textwrap.wrap(right, width - len(block),
                                           break_on_hyphens=False)]


def render_text(sections=None):
    sections = SECTIONS if sections is None else sections
    out = [USAGE, ""]
    for _key, title, rows in sections:
        out.append(title.upper() + ":")
        for row in rows:
            left, right, example, _lang = _row(row)
            out.extend(_wrap(left, right))
            # The example is the part that answers "how do I actually call it", so it is
            # indented under the entry and copied verbatim, not wrapped.
            for line in example:
                out.append(("      " + line).rstrip())
        out.append("")
    out.append("EXAMPLES:")
    for e in EXAMPLES:
        out.append("  " + e)
    out.append("")
    out.append("EVERY SECTION BY KEY: " + ", ".join(s[0] for s in SECTIONS))
    out.append("SEE docs_cpu/QUICKSTART.md AND docs_cpu/API_REFERENCE.md FOR MORE.")
    return "\n".join(out)


def render_markdown(sections=None):
    sections = SECTIONS if sections is None else sections
    out = ["# CPU fork reference", "",
           f"`{USAGE.splitlines()[0].strip()}`", "",
           "Generated by `bitsandbytes-cpu help --markdown`; edit "
           "`bitsandbytes_cpu_cli.py` instead of this file.", ""]
    for _key, title, rows in sections:
        out += [f"## {title}", ""]
        for row in rows:
            left, right, example, lang = _row(row)
            out.append(f"**`{left.replace(chr(10) + ' ', ' ')}`** -- {right}")
            out.append("")
            if example:
                fence = "python" if lang == PY else lang
                out += [f"```{fence}"] + example + ["```", ""]
    out += ["## Examples", "", "```bash"]
    out += EXAMPLES
    out += ["```", ""]
    out += ["## Section keys", "",
            "`bitsandbytes-cpu help <key>` prints one section:", "",
            "`" + "`, `".join(s[0] for s in SECTIONS) + "`", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------
# Reaching into the package without importing it
#
# `import bitsandbytes` runs bitsandbytes/__init__.py, which imports torch. On a machine
# that has this package but not torch that raises, so nothing below uses it: the package
# directory is located with find_spec (which does not execute the package) and the two
# modules that are stdlib-only are loaded from their paths.
# ---------------------------------------------------------------------------------


def _package_dir():
    """Path of the installed bitsandbytes package, without importing it."""
    spec = importlib.util.find_spec("bitsandbytes")
    if spec is not None and spec.submodule_search_locations:
        return list(spec.submodule_search_locations)[0]
    # Running from a source checkout with no metadata installed.
    here = os.path.dirname(os.path.abspath(__file__))
    guess = os.path.join(here, "bitsandbytes")
    return guess if os.path.isdir(guess) else None


def _load_from_package(name, alias=None):
    """Load `bitsandbytes/<name>.py` as a top-level module, bypassing __init__.py."""
    root = _package_dir()
    if root is None:
        raise ImportError("the bitsandbytes package directory was not found")
    path = os.path.join(root, name + ".py")
    if not os.path.isfile(path):
        raise ImportError(f"{path} is missing from the installation")
    alias = alias or name
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _detect():
    """detect_cpu.detect(), with no torch required.

    detect_cpu falls back to `import torch_cpu_kit` when the relative import fails, and
    that is the path taken here; loading it under its bare name first is what makes the
    physical-core count right. Without it the module falls through to os.cpu_count(),
    which reports LOGICAL cores -- 12 on a 6C/12T i5-10400 -- and the recommendation this
    command exists to print would be wrong by a factor the user cannot see.
    """
    _load_from_package("torch_cpu_kit")
    return _load_from_package("detect_cpu").detect()


def _torch_hint():
    print("  torch is not installed.")
    print("  This package does not depend on it on purpose: on PyPI the name `torch`")
    print("  is the CUDA build, about 2.5 GB of nvidia-* wheels. Install the CPU one:")
    print()
    print("      pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("  On Windows and Linux the x86-64 CPU build is also published on PyPI as")
    print("  `torch` for some versions; the index above is the one that always is CPU.")


# ---------------------------------------------------------------------------------


def cmd_help(args) -> int:
    sections, err = _select(getattr(args, "section", None))
    if err is not None:
        print(f"  no section matches {err!r}")
        print("  sections: " + ", ".join(s[0] for s in SECTIONS))
        return 1
    print(render_markdown(sections) if args.markdown else render_text(sections))
    return 0


def cmd_version(args) -> int:
    print(f"  bitsandbytes-cpu-fork : {_safe(_pkg_version)}")
    print(f"  python                : {platform.python_version()}")
    print(f"  platform              : {platform.platform()}")
    print(f"  torch                 : {_safe(lambda: __import__('torch').__version__)}")
    print(f"  torch cuda available  : {_safe(lambda: __import__('torch').cuda.is_available())}")
    print(f"  package path          : {_package_dir() or '<not found>'}")
    return 0


def cmd_detect(args) -> int:
    import json as _json
    try:
        d = _detect()
    except ImportError as exc:
        print(f"  could not read the CPU: {exc}")
        return 1
    if args.json:
        print(_json.dumps(d, ensure_ascii=False, indent=2))
        return 0
    print(f"  CPU        : {d['cpu_model']}")
    print(f"  cores      : {d['physical_cores']} physical / {d['logical_cores']} logical")
    cap = d["cpu_capability"]
    # Without torch there is nothing to ask about SIMD, and "unknown" on its own reads
    # like a defect in this command rather than a missing dependency.
    print(f"  SIMD       : {cap}"
          + ("  (install torch to read it)" if cap in ("unknown", "DEFAULT") else ""))
    if d["ram_total_gb"]:
        print(f"  memory     : {d['ram_total_gb']} GB total, "
              f"{d['ram_available_gb']} GB free")
    else:
        # _mem() returns zeros when psutil is absent, and a "0.0 GB / 0.0 GB" line reads
        # as a machine out of memory rather than a missing dependency.
        print("  memory     : unavailable (pip install psutil)")
    print(f"  threads    : {d['recommend_threads']['lm']} for LLM, "
          f"{d['recommend_threads']['diffusion']} for diffusion")
    print(f"  bf16       : {d['recommend_bf16']}")
    print(f"  8-bit opt  : {d['recommend_8bit_optimizer']}")
    print(f"  disk bal   : {d['recommend_disk_balancer']}")
    return 0


def cmd_doctor(args) -> int:
    import ctypes
    import glob
    pkg = _package_dir()
    if pkg is None:
        print("  bitsandbytes is not installed anywhere importable")
        return 1
    print(f"  package      {pkg}")
    found = glob.glob(os.path.join(pkg, "libbitsandbytes*"))
    print("  files in the package")
    for p in found:
        print(f"    {os.path.basename(p):<28} {os.path.getsize(p)/1e6:.2f} MB")
    if not found:
        print("    none")
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    target = os.path.join(pkg, f"libbitsandbytes_cpu{suffix}")
    print(f"  this platform wants  {os.path.basename(target)}")
    if not os.path.isfile(target):
        print("    MISSING -- install from a wheel for this platform, or from the sdist")
        print("    which compiles it (Linux needs g++ and libomp).")
        return 1
    try:
        lib = ctypes.CDLL(target)
        print("    loads OK")
    except OSError as exc:
        print(f"    FAILED to load: {exc}")
        return 1
    print("  symbols")
    missing = 0
    for n in ("cdequantize_blockwise_cpu_fp32", "cquantize_blockwise_cpu_fp32",
              "cgemv_4bit_inference_cpu_fp32", "cgemm_8bit_inference_cpu_fp32",
              "coptimizer_update_8bit_blockwise_cpu"):
        ok = hasattr(lib, n)
        missing += 0 if ok else 1
        print(f"    {'yes' if ok else 'NO '}  {n}")
    print(f"    {'yes' if hasattr(lib, 'has_avx512bf16_cpu') else 'no '}  "
          f"has_avx512bf16_cpu  (optional: AVX512-BF16 builds only)")
    return 1 if missing else 0


def cmd_selftest(args) -> int:
    try:
        import torch
        import torch.nn as nn
        import bitsandbytes as bnb
    except ImportError:
        _torch_hint()
        return 1

    fails = 0
    print("  [1/3] 4-bit layer against fp32")
    torch.manual_seed(0)
    lin = nn.Linear(512, 256, bias=True)
    q = bnb.nn.Linear4bit(512, 256, bias=True, compute_dtype=torch.float32,
                          quant_type="nf4", compress_statistics=True)
    q.weight = bnb.nn.Params4bit(lin.weight.data, requires_grad=False,
                                 quant_type="nf4", compress_statistics=True)
    q.weight = q.weight.to("cpu")
    q.bias = nn.Parameter(lin.bias.detach().clone())
    x = torch.randn(8, 512)
    y, ref = q(x), lin(x)
    rel = ((y - ref).norm() / ref.norm()).item()
    ok = bool(torch.isfinite(y).all()) and 0.02 < rel < 0.30
    print(f"        relative error {rel:.4f}  (nf4 normally 0.05-0.15)  "
          f"{'OK' if ok else 'FAILED'}")
    fails += 0 if ok else 1

    print("  [2/3] 8-bit optimizer")
    try:
        opt = bnb.optim.AdamW8bit([nn.Parameter(torch.randn(8192))], lr=1e-3)
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        before = p.detach().clone()
        opt.step()
        moved = (p.detach() - before).abs().max().item()
        ok = moved > 0 and bool(torch.isfinite(p).all())
        print(f"        parameter moved by {moved:.3e}  {'OK' if ok else 'FAILED'}")
        fails += 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"        FAILED {type(exc).__name__}: {exc}")
        fails += 1

    print("  [3/3] fused Gated DeltaNet kernel")
    try:
        try:
            from . import gdn_cpu
        except ImportError:
            from bitsandbytes import gdn_cpu
        gdn_cpu.load_native()
        print("        native kernel loaded  OK")
    except Exception as exc:  # noqa: BLE001
        print(f"        FAILED {type(exc).__name__}: {exc}")
        fails += 1

    print(f"\n  {fails} failure(s)")
    return 1 if fails else 0


def _safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


def _pkg_version():
    """The installed version, from distribution metadata.

    Not `bitsandbytes.__version__`: that import needs torch, and `version` is one of the
    commands that has to work without it.
    """
    try:
        from importlib.metadata import version
        return version("bitsandbytes-cpu-fork")
    except Exception:  # noqa: BLE001  (not installed; running from a checkout)
        return "<not installed>"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bitsandbytes-cpu", add_help=False,
        description="Reference and diagnostics for the bitsandbytes CPU fork.")
    ap.add_argument("-h", "--help", action="store_true",
                    help="same as `help`")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("help", add_help=False)
    p.add_argument("section", nargs="?", default=None,
                   help="one section key, e.g. 8bitopt (default: everything)")
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(func=cmd_help)
    d = sub.add_parser("detect", add_help=False)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_detect)
    sub.add_parser("doctor", add_help=False).set_defaults(func=cmd_doctor)
    sub.add_parser("selftest", add_help=False).set_defaults(func=cmd_selftest)
    sub.add_parser("version", add_help=False).set_defaults(func=cmd_version)

    a, _ = ap.parse_known_args(argv)
    if getattr(a, "help", False) or not getattr(a, "func", None):
        return cmd_help(argparse.Namespace(markdown=False, section=None))
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
