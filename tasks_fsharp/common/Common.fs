module TasksFSharp.Common

open System
open Brahma.FSharp

let mib = 1024.0 * 1024.0

let private bytesToMib (bytes: int64<'u>) =
    float (int64 bytes) / mib

let getDevices () =
    ClDevice.GetAvailableDevices(Platform.Any, DeviceType.Default)
    |> Seq.toArray

let tryGetFirstDevice () =
    getDevices () |> Array.tryHead

let getRuntimeContext () =
    match tryGetFirstDevice () with
    | Some d -> RuntimeContext(d), d
    | None -> failwith "No OpenCL device found."

let printDevice (index: int) (d: ClDevice) =
    let maxItems =
        d.MaxWorkItemSizes
        |> Array.map string
        |> String.concat " x "

    printfn "[Device %d] %s" index d.Name
    printfn "  Platform: %A" d.Platform
    printfn "  Type: %A" d.DeviceType
    printfn "  Global memory: %.2f MiB" (bytesToMib d.GlobalMemSize)
    printfn "  Local memory: %.2f KiB" (float (int64 d.LocalMemSize) / 1024.0)
    printfn "  Max work-group size: %d" d.MaxWorkGroupSize
    printfn "  Max work-item dimensions: %d" d.MaxWorkItemDimensions
    printfn "  Max work-item sizes: %s" maxItems
    printfn "  Extensions count: %d" d.DeviceExtensions.Length

let printAllDevices () =
    let devices = getDevices ()

    if devices.Length = 0 then
        printfn "No OpenCL devices found."
    else
        printfn "OpenCL devices found: %d" devices.Length
        devices |> Array.iteri printDevice

let runOpenCl (context: RuntimeContext) computation =
    computation |> ClTask.runSync context
