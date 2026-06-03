# Lorenz CPU vs OpenCL (NVIDIA GPU)

This project compares a high-precision Lorenz attractor simulation on:

- CPU (C++, double precision)
- OpenCL GPU (C++ API via `CL/cl.hpp`, double precision)

Both paths use the same ODE algorithm: classical RK4 with identical equations and parameter order.

## Build

From repo root:

```powershell
cmake -S . -B build
cmake --build build --target lorenz_cpu_vs_opencl --config Debug
```

## Run

From repo root:

```powershell
.\build\lorenz_comparison\Debug\lorenz_cpu_vs_opencl.exe [steps] [trajectories]
```

Example:

```powershell
.\build\lorenz_comparison\Debug\lorenz_cpu_vs_opencl.exe 4000 2048
```

## Output

- Console:
  - Selected OpenCL device (prefers NVIDIA GPU when available)
  - CPU and OpenCL timings
  - Speedup
  - Max absolute differences
- CSV file: `lorenz_validation.csv` (written to current working directory)

CSV columns:

- `step,time`
- `cpu_x,cpu_y,cpu_z`
- `gpu_x,gpu_y,gpu_z`
- `abs_dx,abs_dy,abs_dz,max_abs_diff`

The CSV is used to validate that CPU and OpenCL computations are numerically consistent.
