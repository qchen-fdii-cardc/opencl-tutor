# tasks_fsharp

This folder provides an F# OpenCL learning track implemented with Brahma.FSharp.

The structure mirrors the C/C++ tasks in this repository:
- 00: device report
- 01-14: progressive compute tasks (inventory, queues, kernels, timing, memory, matrix math, local memory, benchmarking, extensions, non-graphics compute, and multi-device execution)

## Why Brahma.FSharp

Brahma.FSharp lets you write kernels as F# quotations and execute them on OpenCL devices while keeping host-side logic in idiomatic F#.

Core pattern used across tasks:
1. Select device and build RuntimeContext.
2. Define quotation kernel (Range1D + clarray arguments).
3. Copy host arrays to device with ClArray.toDevice.
4. Execute with runCommand and a configured NDRange.
5. Copy results back with ClArray.toHost and validate.

## Solution Layout

- OpenCL.Tasks.FSharp.slnx: top-level .NET solution for all tasks.
- common/Common.fs: device discovery, context setup, and reusable reporting helpers.
- common/TaskRunners.fs: implementation of all task runners.
- 00_device_report ... 14_multi_device_parallel: per-task executable entrypoints and task docs.

## Build And Run

```powershell
dotnet build .\OpenCL.Tasks.FSharp.slnx
```

Run one task (example: task 03):

```powershell
dotnet run --project .\03_vector_add_kernel\03_vector_add_kernel.fsproj
```

Run device report:

```powershell
dotnet run --project .\00_device_report\00_device_report.fsproj
```

## Task Index

| Task | Folder | Runner Function |
|---|---|---|
| 00 | 00_device_report | run00DeviceReport |
| 01 | 01_platform_device_inventory | run01PlatformDeviceInventory |
| 02 | 02_context_and_queue | run02ContextAndQueue |
| 03 | 03_vector_add_kernel | run03VectorAddKernel |
| 04 | 04_error_handling | run04ErrorHandling |
| 05 | 05_kernel_timing | run05KernelTiming |
| 06 | 06_memory_model_buffers | run06MemoryModelBuffers |
| 07 | 07_matrix_multiply | run07MatrixMultiply |
| 08 | 08_workgroup_local_memory | run08WorkgroupLocalMemory |
| 09 | 09_cpu_vs_opencl_benchmark | run09CpuVsOpenClBenchmark |
| 10 | 10_images_and_samplers | run10ImagesAndSamplers |
| 11 | 11_gaussian_blur | run11GaussianBlur |
| 12 | 12_extensions | run12Extensions |
| 13 | 13_nongraphics_apps | run13NonGraphicsApps |
| 14 | 14_multi_device_parallel | run14MultiDeviceParallel |

## Notes On Current Toolchain

- Projects currently target net7.0 for Brahma.FSharp compatibility in this workspace.
- On newer preview SDKs, you may see EOL warnings for net7.0 and dependency constraint warnings (NU1608). Tasks still build and run.

## Brahma.FSharp Resources

- GitHub: https://github.com/YaccConstructor/Brahma.FSharp
- README quick start: https://github.com/YaccConstructor/Brahma.FSharp#quick-start
- Basic example article: https://yaccconstructor.github.io/Brahma.FSharp/Articles/Basic_Example.html
- API reference landing: https://yaccconstructor.github.io/Brahma.FSharp/
