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
# What remains is the one thing that still matters: telling the wheel machinery
# that this distribution is NOT pure Python. Without that, the wheel would be
# tagged py3-none-any -- claiming to be installable on Linux and macOS -- while
# containing a Windows DLL. With it, the tag becomes cp3XX-cp3XX-win_amd64, so pip
# refuses it on any other platform instead of installing a package whose kernel
# cannot load.
#
# The upstream version is kept beside this file as setup.upstream-cuda.py.bak.
# ---------------------------------------------------------------------------
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Mark the distribution as platform-specific.

    Setuptools derives the wheel tag from this. A DLL is not a Python extension, so
    nothing else in the build would mark it.
    """

    def has_ext_modules(self):
        return True


setup(
    version="0.50.2.dev0",
    distclass=BinaryDistribution,
)
