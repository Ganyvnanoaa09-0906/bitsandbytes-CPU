# 本机 native 编译工具链（zig）

- zig 0.14.1：`D:\work\tools-zig\zig\zig-x86_64-windows-0.14.1\zig.exe`（自带 clang 前端，无需 MSVC）
- 编译 AVX2 DLL：
  ```powershell
  & 'D:\work\tools-zig\zig\zig-x86_64-windows-0.14.1\zig.exe' c++ -O3 -mavx2 -mfma -shared `
    gt1030/q2bit_gemv.cpp -o gt1030/q2bit_gemv.dll
  ```
- Python 侧 `ctypes.CDLL` 加载；导出用 `extern "C" __declspec(dllexport)`。
- 已有示例：`gt1030/q2bit_gemv.cpp`（2bit uniform LUT fused GEMV）。
- 注意：绝对路径，zig 不在 PATH；DLL 常被 .gitignore 忽略，提交 `.cpp` + 本节命令即可。
