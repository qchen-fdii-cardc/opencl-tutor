# lorenz_comparison_fsharp

F# console app that compares Lorenz attractor integration on:

- CPU (host RK4 in double precision)
- OpenCL via Brahma.FSharp (same RK4 formula in kernel quotation)

It prefers an NVIDIA device when one is present, and outputs a CSV for numerical validation.

## Initialize / Build

Project was initialized with:

```powershell
dotnet new console -lang F# -n lorenz_comparison_fsharp
```

Build:

```powershell
dotnet build .\lorenz_comparison\lorenz_comparison_fsharp\lorenz_comparison_fsharp.fsproj
```

## Run

```powershell
dotnet run --project .\lorenz_comparison\lorenz_comparison_fsharp\lorenz_comparison_fsharp.fsproj -- [steps] [trajectories]
```

Example:

```powershell
dotnet run --project .\lorenz_comparison\lorenz_comparison_fsharp\lorenz_comparison_fsharp.fsproj -- 2000 1024
```

## Output

- Console report:
  - Selected OpenCL device
  - CPU and OpenCL elapsed time
  - Speedup ratio
  - Max absolute differences (final states and full history)
- CSV:
  - `lorenz_comparison/lorenz_comparison_fsharp/lorenz_validation_fsharp.csv`
  - Columns match the C++ comparison style:
    `step,time,cpu_x,cpu_y,cpu_z,gpu_x,gpu_y,gpu_z,abs_dx,abs_dy,abs_dz,max_abs_diff`
