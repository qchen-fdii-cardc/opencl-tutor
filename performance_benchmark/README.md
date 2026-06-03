# OpenCL Multi-Device Stress Benchmark

This folder contains a C++ OpenCL benchmark that uses all detected OpenCL devices concurrently.

## What It Does

- Enumerates all OpenCL platforms and devices.
- Starts one worker thread per device.
- Runs a heavy math kernel in a loop for a configurable duration.
- Uses `-cl-fast-relaxed-math` to increase arithmetic throughput.

The goal is sustained high compute load for thermal and power stress testing.

## Build

From the repository root:

```powershell
cmake -S . -B build
cmake --build build --target opencl_multi_device_stress --config Debug
```

## Run

From the repository root:

```powershell
.\build\performance_benchmark\Debug\opencl_multi_device_stress.exe 30
```

The argument is the run duration in seconds. Press `Ctrl+C` to stop early.

## Notes

- Start with short runs (`5-15` seconds) and monitor temperatures.
- Longer runs can significantly increase heat and fan speed.
- Throughput numbers are estimates for quick comparison between devices.
