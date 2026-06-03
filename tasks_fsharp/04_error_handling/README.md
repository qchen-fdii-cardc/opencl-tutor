# 04 Error Handling

## Goal

Demonstrate expected failures and recovery path handling.

## Implementation In This Repository

- Runner function: run04ErrorHandling in common/TaskRunners.fs.
- Entry point: Program.fs calls the runner directly.
- Device/context helpers: common/Common.fs.

## Key Brahma.FSharp Concepts Used

Graceful device-selection failure, try/with around OpenCL execution.

Frequently used APIs in this task set:
- ClDevice.GetAvailableDevices / ClDevice.GetFirstAppropriateDevice
- RuntimeContext
- opencl computation expression
- ClArray.toDevice / ClArray.alloc / ClArray.toHost
- runCommand
- Range1D and Range1D.CreateValid

## Run

`powershell
dotnet run --project ./04_error_handling/04_error_handling.fsproj
`

## Expected Output

Captured expected selection error plus recovery execution result.

## Troubleshooting

- If no device is detected, verify OpenCL runtime/driver installation.
- If build warns about target framework, this track intentionally uses net7.0 for current Brahma package compatibility.
- If output looks unexpected, run 00_device_report first to verify actual selected device.

## References

- Brahma.FSharp Quick Start: https://github.com/YaccConstructor/Brahma.FSharp#quick-start
- Basic Example: https://yaccconstructor.github.io/Brahma.FSharp/Articles/Basic_Example.html
