# pso-opencl

OpenCL-accelerated PSO framework in C++ using the OpenCL C++ API (`CL/cl.hpp`).

## Features

- GPU-accelerated particle fitness evaluation (Rastrigin objective).
- GPU-accelerated velocity/position update.
- NVIDIA GPU preference when available.
- Iteration-by-iteration best-fitness convergence exported to CSV.
- Reusable framework class (`OpenClPsoFramework`) for extending objectives and update logic.

## Build

From repository root:

```powershell
cmake -S . -B build
cmake --build build --target pso_opencl --config Release
```

## Run

From repository root:

```powershell
.\build\pso-opencl\Release\pso_opencl.exe [particles] [dimensions] [iterations]
```

Example:

```powershell
.\build\pso-opencl\Release\pso_opencl.exe 8192 32 400
```

## Output

- Console summary:
  - selected device
  - run size (particles, dimensions, iterations)
  - best objective value
  - elapsed time
  - particle evaluations per second
- CSV file:
  - `pso-opencl/pso_convergence.csv`
  - columns: `iteration,best_fitness`
