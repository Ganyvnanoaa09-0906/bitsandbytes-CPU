# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE file in
# the root directory of this source tree.
#
# ---------------------------------------------------------------------------
# CPU-fork packaging notes
#
# The inherited setup.py ran CMake through scikit-build-core to compile the CUDA
# kernels. The Windows artefact of this fork is a hand-built CPU DLL that already
# lives in the package directory, so there is nothing for CMake to do, and the two
# build dependencies it needs are not required by anyone installing a wheel.
#
# TWO TAGS HAVE TO BE RIGHT, AND THEY PULL IN OPPOSITE DIRECTIONS
#
#   plat_name   must be concrete (win_amd64). A DLL is not a Python extension, so
#               left alone the build would tag the wheel py3-none-any -- claiming
#               to be installable on Linux and macOS -- and pip would install it
#               there and then fail at import.
#
#   python_tag  must be py3, not cp311. The load-bearing question is whether the
#               artefact is built against CPython's C API. This one is not: it is a
#               C-ABI DLL loaded through ctypes, so the .pyd ABI rules do not apply.
#               Tagging it cp311-cp311 would demand a separate wheel for every
#               interpreter version, five files where one is correct.
#
# How the tag is produced, read out of setuptools/command/bdist_wheel.py:
#
#     if self.root_is_pure:                      # -> python_tag / none / plat
#         impl = self.python_tag
#     else:                                      # -> cp311 / cp311 / plat
#         impl = tags.interpreter_name() + tags.interpreter_version()
#
# so `root_is_pure` selects the branch, and the non-pure branch IGNORES
# self.python_tag entirely -- assigning python_tag and abi_tag on a non-pure
# distribution (the first version of this file) silently did nothing and still
# produced cp311-cp311-win_amd64.
#
# The distribution is therefore declared pure, which is accurate: no ext_modules
# are built. The platform is fixed in finalize_options, where plat_name_supplied
# is computed from the value just set; setting it afterwards leaves the flag False
# and the tag falls back to "any".
#
# The upstream version is kept beside this file as setup.upstream-cuda.py.bak.
# ---------------------------------------------------------------------------
import os
import platform as _platform
import subprocess

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.dist import Distribution

_HERE = os.path.dirname(os.path.abspath(__file__))


def _native_library_name():
    system = _platform.system()
    if system == "Windows":
        return "libbitsandbytes_cpu.dll"
    if system == "Darwin":
        return "libbitsandbytes_cpu.dylib"
    return "libbitsandbytes_cpu.so"


def _ensure_native_library():
    """Refuse to build a wheel that cannot import.

    The native artefact is an opaque C library loaded through ctypes, and only the
    Windows build is committed. On Linux or macOS the sdist therefore compiles to a
    wheel whose tag says linux_x86_64, installs without a single warning, and then dies
    at `import bitsandbytes` because the .so is not there -- the package data contains a
    Windows DLL. That is a bad failure: every step of the install path reports success.

    So on a platform whose binary is absent, build it if the documented toolchain is
    available, and otherwise stop here with the exact command to run. A build failure is
    strictly better than a wheel that cannot import, because it names the missing piece
    while the user is still looking at pip output.
    """
    lib = os.path.join(_HERE, "bitsandbytes", _native_library_name())
    if os.path.isfile(lib):
        return

    if _platform.system() == "Windows":
        raise RuntimeError(
            "bitsandbytes/libbitsandbytes_cpu.dll is not in this source tree, so the "
            "Windows wheel cannot be built. Rebuild it with build_manual\\build_manual.bat "
            "(see docs_cpu) and try again."
        )

    script = os.path.join(_HERE, "build_linux.sh")
    if not os.path.isfile(script):
        raise RuntimeError(
            f"{_native_library_name()} is missing and build_linux.sh is not in this "
            "source tree either, so there is nothing to build it with. Install from the "
            "wheel for your platform instead of from the sdist."
        )

    print(f"    {_native_library_name()} not found; running build_linux.sh", flush=True)
    rc = subprocess.call(["bash", script], cwd=_HERE)
    if rc != 0 or not os.path.isfile(lib):
        raise RuntimeError(
            f"build_linux.sh exited {rc} and {_native_library_name()} is still missing. "
            "It needs a C++ compiler with OpenMP:\n"
            "    Debian/Ubuntu : sudo apt install g++ libomp-dev\n"
            "    Fedora/RHEL   : sudo dnf install gcc-c++ libomp-devel\n"
            "    Arch          : sudo pacman -S gcc openmp\n"
            "Then run pip install again."
        )
    print(f"    built {lib}", flush=True)


class PureDistribution(Distribution):
    """No ext_modules are built; the native artefact is package data."""

    def has_ext_modules(self):
        return False


class PlatformWheel(_bdist_wheel):
    """py3-none-<platform>: one wheel per platform, not per interpreter."""

    def run(self):
        # Before any wheel is assembled. A wheel that installs but cannot import is the
        # one outcome this must not produce.
        _ensure_native_library()
        super().run()

    def finalize_options(self):
        self.plat_name = _wheel_platform()
        super().finalize_options()
        # root_is_pure is what routes get_tag() to the python_tag branch.
        self.root_is_pure = True
        self.python_tag = "py3"


def _wheel_platform():
    """The wheel platform tag for the machine doing the build.

    Only the platform this is built on can be produced here, because the DLL in
    package-data is a Windows binary; a Linux build of the same source emits
    linux_x86_64 through the same call.
    """
    machine = _platform.machine().lower()
    if _platform.system() == "Windows":
        return "win_amd64" if machine in ("amd64", "x86_64") else "win32"
    if _platform.system() == "Darwin":
        return "macosx_11_0_arm64" if machine == "arm64" else "macosx_10_9_x86_64"
    # Linux and everything else
    return "linux_aarch64" if machine in ("aarch64", "arm64") else "linux_x86_64"


setup(
    version="0.50.2.dev0",
    distclass=PureDistribution,
    cmdclass={"bdist_wheel": PlatformWheel},
)
