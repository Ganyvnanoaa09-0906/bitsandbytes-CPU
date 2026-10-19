# Third-party notices

This Windows package bundles / depends on the following third-party software.

## CLBlast (bundled: `clblast.dll`)

- Project: https://github.com/CNugteren/CLBlast
- License: Apache License, Version 2.0
- Copyright (c) 2015, 2016, 2017, 2018, 2019, 2020 Cedric Nugteren
- The full license text is included as `CLBlast-LICENSE.txt`.

## Python packages installed by `setup.bat` via pip (not bundled)

| Package | License |
|---|---|
| torch (PyTorch) | BSD-3-Clause |
| transformers | Apache-2.0 |
| numpy | BSD-3-Clause |
| pyopencl | MIT |
| safetensors | Apache-2.0 |
| psutil | BSD-3-Clause |

Their licenses are distributed inside their own wheels by pip.

## OpenCL

OpenCL is provided by the GPU driver installed on the machine (AMD / Intel / NVIDIA).
This package does not bundle an OpenCL ICD or driver.
