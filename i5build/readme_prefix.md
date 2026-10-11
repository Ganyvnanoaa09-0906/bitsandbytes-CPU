> **This is the CPU fork** — `bitsandbytes-cpu-fork`, not upstream `bitsandbytes`.
> It replaces the CUDA kernels with a pure-CPU backend (AVX2 on x86-64, NEON on ARM64):
> 8-bit optimizers, fused Gated DeltaNet, `gemm_8bit`, block-wise 4/8-bit quantisation.
> No GPU, no CUDA toolkit, no NVIDIA driver. Full guide:
> [`docs_cpu/QUICKSTART.md`](docs_cpu/QUICKSTART.md) (中文) ·
> [`docs_cpu/QUICKSTART_EN.md`](docs_cpu/QUICKSTART_EN.md).

## Install

**`torch` is deliberately not a dependency.** On PyPI the name `torch` resolves to the
CUDA build, so declaring it made this CPU-only library download roughly 2.5 GB of
`nvidia-*` wheels — cublas, cudnn, nccl, nvshmem and thirteen more — onto machines with
no NVIDIA device at all. Measured on a clean Ubuntu install. `install_requires` cannot
express "the CPU build", because that lives on a different index, so install it first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install bitsandbytes-cpu-fork
```

### Windows

A wheel is published with the CPU library already built in, so no compiler and no
Visual C++ Redistributable are needed:

```bat
pip install bitsandbytes-cpu-fork
```

### Linux

Only the source distribution is published. `pip` builds it, and `setup.py` runs
`build_linux.sh` for you, so the library is compiled against your own glibc rather than
shipped prebuilt. That needs a C++ compiler with OpenMP:

```bash
sudo apt install g++ libomp-dev        # Debian / Ubuntu
sudo dnf install gcc-c++ libomp-devel  # Fedora / RHEL
sudo pacman -S gcc openmp              # Arch
```

If the toolchain is missing the build **stops and names that command**, rather than
producing a wheel that installs cleanly and then fails at `import bitsandbytes`.

To build by hand instead, or to check the kernels without torch:

```bash
bash build_linux.sh --selftest
```

### Verify the install

```bash
python -m bitsandbytes
```

---

