# 02 Context And Queue

## Goal

Create RuntimeContext and verify command queue setup.

## Implementation In This Repository

- Runner function: run02ContextAndQueue in common/TaskRunners.fs.
- Entry point: Program.fs calls the runner directly.
- Device/context helpers: common/Common.fs.

## Key Brahma.FSharp Concepts Used

RuntimeContext lifecycle, selecting a working device.

Frequently used APIs in this task set:
- ClDevice.GetAvailableDevices / ClDevice.GetFirstAppropriateDevice
- RuntimeContext
- opencl computation expression
- ClArray.toDevice / ClArray.alloc / ClArray.toHost
- runCommand
- Range1D and Range1D.CreateValid

## Run

`powershell
dotnet run --project ./02_context_and_queue/02_context_and_queue.fsproj
`

## Expected Output

Selected device plus context initialization message.

## Troubleshooting

- If no device is detected, verify OpenCL runtime/driver installation.
- If build warns about target framework, this track intentionally uses net7.0 for current Brahma package compatibility.
- If output looks unexpected, run 00_device_report first to verify actual selected device.

## References

- Brahma.FSharp Quick Start: https://github.com/YaccConstructor/Brahma.FSharp#quick-start
- Basic Example: https://yaccconstructor.github.io/Brahma.FSharp/Articles/Basic_Example.html
