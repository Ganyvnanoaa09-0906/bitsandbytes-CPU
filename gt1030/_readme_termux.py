# -*- coding: utf-8 -*-
"""README：修坏掉的 TERMUX_GUIDE 链接 + 在 "Build from source" 下补一节 Termux 构建说明。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\README.md'
s = io.open(p, encoding='utf-8').read()

# 1) 修坏链接（原指 docs_cpu/TERMUX_GUIDE.md，实际文件在 bitsandbytes/TERMUX_GUIDE.md）
old = "- **aarch64**: there is no PyPI wheel yet; build from the sdist (`build_linux.sh`) or use the\n  Termux instructions in `docs_cpu/TERMUX_GUIDE.md`."
new = ("- **aarch64**: there is no PyPI wheel yet; build from the sdist (`build_linux.sh`) or follow the\n"
       "  Termux guide [`TERMUX_GUIDE.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/TERMUX_GUIDE.md).")
assert old in s, 'aarch64 段落没匹配上'
s = s.replace(old, new, 1)

# 2) 在 "Build from source" 的构建块之后插入 Termux 小节
anchor = """Two special wheel builds used by this project are documented in"""
termux = """### Termux (Android / ARM64)

On-device builds work; there is no aarch64 PyPI wheel, so build from the sdist:

```bash
pkg update && pkg install -y clang libomp make python
bash build_termux.sh                 # -> bitsandbytes/libbitsandbytes_cpu.so
bash build_termux.sh --selftest      # also build + run the torch-free C self-test
```

Three Termux-specific traps the script already handles: the compiler is `clang++` (there is no
`g++`), OpenMP needs `pkg install libomp`, and `-march=native` is unsafe under Android's clang.
The script is also what the automated flow uses to build on a phone **driven from a PC over ADB**
— see [`TERMUX_GUIDE.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/TERMUX_GUIDE.md)
for the phone-side preparation, the ADB/wireless-debugging path, the ARM64 kernel notes and the
verification matrix.

"""
assert anchor in s
s = s.replace(anchor, termux + anchor, 1)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('README 已更新：Termux 小节 + 修链接')
