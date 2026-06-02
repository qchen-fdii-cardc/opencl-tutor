# Minimal OpenCL + CMake Example

This project builds a tiny C++ executable that:
- finds the first OpenCL platform,
- finds one device on that platform,
- creates an OpenCL context,
- prints the selected device name.

## Requirements

- CMake 3.16+
- A C++ compiler (MSVC, Clang, or GCC)
- OpenCL runtime + development files (headers and library)

On Windows, this usually comes from your GPU driver (NVIDIA/AMD/Intel). If CMake cannot find OpenCL, install the vendor SDK/runtime and ensure the OpenCL library is available to your toolchain.

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Run

```powershell
.\build\Release\opencl_minimal.exe
```

For single-config generators (for example Ninja), run:

```powershell
.\build\opencl_minimal.exe
```
