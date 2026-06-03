module TasksFSharp.TaskRunners

open System
open System.Diagnostics
open Brahma.FSharp
open TasksFSharp.Common

let private defaultWg = 256

let private validRange n =
    let wg = min defaultWg (max 1 n)
    Range1D.CreateValid(n, wg)

let run00DeviceReport () =
    printAllDevices ()

let run01PlatformDeviceInventory () =
    printAllDevices ()

let run02ContextAndQueue () =
    let context, device = getRuntimeContext ()
    printfn "Selected device: %s" device.Name
    printfn "Runtime context initialized: %O" context

let run03VectorAddKernel () =
    let context, device = getRuntimeContext ()
    printfn "Device: %s" device.Name

    let n = 1 <<< 20
    let a = Array.init n (fun i -> float32 (i % 131))
    let b = Array.init n (fun i -> float32 (i % 47))

    let kernel =
        <@
            fun (range: Range1D) (x: float32 clarray) (y: float32 clarray) (z: float32 clarray) ->
                let i = range.GlobalID0
                if i < n then
                    z.[i] <- x.[i] + y.[i]
        @>

    let gpu =
        opencl {
            use! x = ClArray.toDevice a
            use! y = ClArray.toDevice b
            use! z = ClArray.alloc<float32> n
            do! runCommand kernel <| fun k -> k <| validRange n <| x <| y <| z
            return! ClArray.toHost z
        }
        |> runOpenCl context

    let ok =
        gpu
        |> Array.mapi (fun i v -> abs (v - (a.[i] + b.[i])) < 1e-5f)
        |> Array.forall id

    printfn "%s" (if ok then "Vector add verified." else "Vector add failed.")

let run04ErrorHandling () =
    try
        ClDevice.GetFirstAppropriateDevice(platform = Platform.Custom "DefinitelyMissingPlatform*") |> ignore
        printfn "Unexpected: custom platform query succeeded."
    with ex ->
        printfn "Captured expected selection error: %s" ex.Message

    try
        let context, _ = getRuntimeContext ()
        let kernel = <@ fun (r: Range1D) (buf: int clarray) -> buf.[r.GlobalID0] <- buf.[r.GlobalID0] + 1 @>
        let result =
            opencl {
                use! buf = ClArray.toDevice [| 1; 2; 3; 4 |]
                do! runCommand kernel <| fun k -> k <| validRange 4 <| buf
                return! ClArray.toHost buf
            }
            |> runOpenCl context
        printfn "Recovery run OK: %A" result
    with ex ->
        printfn "Unexpected runtime error: %s" ex.Message

let run05KernelTiming () =
    let context, device = getRuntimeContext ()
    printfn "Device: %s" device.Name

    let n = 1 <<< 20
    let data = Array.create n 1.0f

    let kernel =
        <@
            fun (range: Range1D) (x: float32 clarray) alpha ->
                let i = range.GlobalID0
                if i < n then
                    x.[i] <- x.[i] * alpha
        @>

    let sw = Stopwatch.StartNew()
    let result =
        opencl {
            use! x = ClArray.toDevice data
            do! runCommand kernel <| fun k -> k <| validRange n <| x <| 2.0f
            return! ClArray.toHost x
        }
        |> runOpenCl context
    sw.Stop()

    let ok = result |> Array.forall (fun v -> abs (v - 2.0f) < 1e-6f)
    printfn "Kernel elapsed: %.3f ms" sw.Elapsed.TotalMilliseconds
    printfn "%s" (if ok then "Timing task verified." else "Timing task failed.")

let run06MemoryModelBuffers () =
    let context, _ = getRuntimeContext ()
    let n = 1 <<< 16
    let a = 2.0f
    let x = Array.init n (fun i -> float32 i)
    let y = Array.init n (fun i -> float32 (2 * i))

    let kernel =
        <@
            fun (range: Range1D) alpha (x: float32 clarray) (y: float32 clarray) (z: float32 clarray) ->
                let i = range.GlobalID0
                if i < n then
                    z.[i] <- alpha * x.[i] + y.[i]
        @>

    let z =
        opencl {
            use! dx = ClArray.toDevice x
            use! dy = ClArray.toDevice y
            use! dz = ClArray.alloc<float32> n
            do! runCommand kernel <| fun k -> k <| validRange n <| a <| dx <| dy <| dz
            return! ClArray.toHost dz
        }
        |> runOpenCl context

    let ok = z |> Array.mapi (fun i v -> abs (v - (a * x.[i] + y.[i])) < 1e-5f) |> Array.forall id
    printfn "%s" (if ok then "Memory model task verified." else "Memory model task failed.")

let run07MatrixMultiply () =
    let context, _ = getRuntimeContext ()
    let n = 64
    let total = n * n
    let a = Array.init total (fun i -> float32 ((i / n + i % n) % 7))
    let b = Array.init total (fun i -> float32 ((i / n * 3 + i % n) % 11))

    let kernel =
        <@
            fun (range: Range1D) (a: float32 clarray) (b: float32 clarray) (c: float32 clarray) ->
                let gid = range.GlobalID0
                if gid < total then
                    let row = gid / n
                    let col = gid % n
                    let mutable acc = 0.0f
                    for k in 0 .. n - 1 do
                        acc <- acc + a.[row * n + k] * b.[k * n + col]
                    c.[gid] <- acc
        @>

    let cGpu =
        opencl {
            use! da = ClArray.toDevice a
            use! db = ClArray.toDevice b
            use! dc = ClArray.alloc<float32> total
            do! runCommand kernel <| fun k -> k <| validRange total <| da <| db <| dc
            return! ClArray.toHost dc
        }
        |> runOpenCl context

    let cCpu =
        Array.init total (fun gid ->
            let row = gid / n
            let col = gid % n
            let mutable acc = 0.0f
            for k in 0 .. n - 1 do
                acc <- acc + a.[row * n + k] * b.[k * n + col]
            acc)

    let ok = cGpu |> Array.mapi (fun i v -> abs (v - cCpu.[i]) < 1e-3f) |> Array.forall id
    printfn "%s" (if ok then "Matrix multiply verified." else "Matrix multiply failed.")

let run08WorkgroupLocalMemory () =
    let context, _ = getRuntimeContext ()
    let n = 4096
    let input = Array.init n id

    let kernel =
        <@
            fun (range: Range1D) (input: int clarray) (output: int clarray) ->
                let localCache = localArray<int> 256
                let gid = range.GlobalID0
                let lid = range.LocalID0

                localCache.[lid] <- input.[gid]
                barrierLocal ()

                output.[gid] <- localCache.[lid] + 1
        @>

    let gpu =
        opencl {
            use! src = ClArray.toDevice input
            use! dst = ClArray.alloc<int> n
            do! runCommand kernel <| fun k -> k <| Range1D.CreateValid(n, 256) <| src <| dst
            return! ClArray.toHost dst
        }
        |> runOpenCl context

    let ok = gpu |> Array.mapi (fun i v -> v = input.[i] + 1) |> Array.forall id
    printfn "%s" (if ok then "Local-memory task verified." else "Local-memory task failed.")

let run09CpuVsOpenClBenchmark () =
    let context, _ = getRuntimeContext ()
    let n = 1 <<< 20
    let a = Array.init n (fun i -> float32 (i % 257))
    let b = Array.init n (fun i -> float32 (i % 113))

    let cpuSw = Stopwatch.StartNew()
    let cpu = Array.init n (fun i -> a.[i] + b.[i])
    cpuSw.Stop()

    let kernel =
        <@
            fun (range: Range1D) (x: float32 clarray) (y: float32 clarray) (z: float32 clarray) ->
                let i = range.GlobalID0
                if i < n then
                    z.[i] <- x.[i] + y.[i]
        @>

    let gpuSw = Stopwatch.StartNew()
    let gpu =
        opencl {
            use! dx = ClArray.toDevice a
            use! dy = ClArray.toDevice b
            use! dz = ClArray.alloc<float32> n
            do! runCommand kernel <| fun k -> k <| validRange n <| dx <| dy <| dz
            return! ClArray.toHost dz
        }
        |> runOpenCl context
    gpuSw.Stop()

    let ok = gpu |> Array.mapi (fun i v -> abs (v - cpu.[i]) < 1e-5f) |> Array.forall id
    let speedup = cpuSw.Elapsed.TotalMilliseconds / max 0.0001 gpuSw.Elapsed.TotalMilliseconds
    printfn "CPU: %.3f ms" cpuSw.Elapsed.TotalMilliseconds
    printfn "OpenCL: %.3f ms" gpuSw.Elapsed.TotalMilliseconds
    printfn "Speedup: %.2fx" speedup
    printfn "%s" (if ok then "Benchmark result verified." else "Benchmark mismatch.")

let run10ImagesAndSamplers () =
    let context, _ = getRuntimeContext ()
    let w, h = 128, 128
    let n = w * h
    let pixels = Array.init n (fun i -> uint32 (i &&& 0xFF))

    let kernel =
        <@
            fun (range: Range1D) (src: uint32 clarray) (dst: uint32 clarray) ->
                let i = range.GlobalID0
                if i < n then
                    dst.[i] <- 255u - src.[i]
        @>

    let outPixels =
        opencl {
            use! s = ClArray.toDevice pixels
            use! d = ClArray.alloc<uint32> n
            do! runCommand kernel <| fun k -> k <| validRange n <| s <| d
            return! ClArray.toHost d
        }
        |> runOpenCl context

    let ok = outPixels |> Array.mapi (fun i v -> v = 255u - pixels.[i]) |> Array.forall id
    printfn "%s" (if ok then "Image-like buffer task verified." else "Image-like buffer task failed.")

let run11GaussianBlur () =
    let context, _ = getRuntimeContext ()
    let w, h = 64, 64
    let n = w * h
    let src = Array.init n (fun i -> float32 ((i * 37) % 255))

    let kernel =
        <@
            fun (range: Range1D) (src: float32 clarray) (dst: float32 clarray) ->
                let gid = range.GlobalID0
                if gid < n then
                    let x = gid % w
                    let y = gid / w

                    let mutable sum = 0.0f
                    let mutable weight = 0.0f

                    for dy in -1 .. 1 do
                        for dx in -1 .. 1 do
                            let xx = max 0 (min (w - 1) (x + dx))
                            let yy = max 0 (min (h - 1) (y + dy))
                            let wg =
                                if dx = 0 && dy = 0 then 4.0f
                                elif dx = 0 || dy = 0 then 2.0f
                                else 1.0f

                            sum <- sum + src.[yy * w + xx] * wg
                            weight <- weight + wg

                    dst.[gid] <- sum / weight
        @>

    let dst =
        opencl {
            use! s = ClArray.toDevice src
            use! d = ClArray.alloc<float32> n
            do! runCommand kernel <| fun k -> k <| validRange n <| s <| d
            return! ClArray.toHost d
        }
        |> runOpenCl context

    let changed = dst |> Array.mapi (fun i v -> abs (v - src.[i]) > 1e-3f) |> Array.exists id
    printfn "%s" (if changed then "Gaussian blur task verified." else "Gaussian blur task failed.")

let run12Extensions () =
    let devices = getDevices ()
    if devices.Length = 0 then
        printfn "No OpenCL devices found."
    else
        devices
        |> Array.iteri (fun i d ->
            printfn "[Device %d] %s" i d.Name
            printfn "  Type: %A" d.DeviceType
            printfn "  Max WG size: %d" d.MaxWorkGroupSize
            printfn "  Extensions:"
            d.DeviceExtensions |> Array.iter (fun ext -> printfn "    - %A" ext)
            printfn "")

let run13NonGraphicsApps () =
    let context, _ = getRuntimeContext ()
    let workItems = 4096
    let samplesPerItem = 1024

    let seeds = Array.init workItems (fun i -> uint32 (1234567 + i * 265443576))

    let kernel =
        <@
            fun (range: Range1D) (seeds: uint32 clarray) (hits: int clarray) samplesPerItem ->
                let gid = range.GlobalID0
                let mutable s = seeds.[gid]
                let mutable count = 0

                for _ in 0 .. samplesPerItem - 1 do
                    s <- 1664525u * s + 1013904223u
                    let x = float32 (int (s &&& 0x00FFFFFFu)) / 16777216.0f
                    s <- 1664525u * s + 1013904223u
                    let y = float32 (int (s &&& 0x00FFFFFFu)) / 16777216.0f
                    if x * x + y * y <= 1.0f then
                        count <- count + 1

                hits.[gid] <- count
        @>

    let hits =
        opencl {
            use! s = ClArray.toDevice seeds
            use! h = ClArray.alloc<int> workItems
            do! runCommand kernel <| fun k -> k <| validRange workItems <| s <| h <| samplesPerItem
            return! ClArray.toHost h
        }
        |> runOpenCl context

    let totalHits = hits |> Array.sum |> float
    let totalSamples = float (workItems * samplesPerItem)
    let pi = 4.0 * totalHits / totalSamples
    let err = abs (pi - Math.PI)

    printfn "Estimated pi: %.8f" pi
    printfn "Absolute error: %.8f" err
    printfn "%s" (if err < 0.05 then "Non-graphics task verified." else "Non-graphics task failed.")

let run14MultiDeviceParallel () =
    let devices = getDevices () |> Array.truncate 2
    if devices.Length = 0 then
        failwith "No OpenCL devices found."

    let n = 1 <<< 20
    let a = Array.init n (fun i -> float32 (i % 97))
    let b = Array.init n (fun i -> float32 (i % 41))
    let output = Array.zeroCreate<float32> n

    let kernel chunkSize =
        <@
            fun (range: Range1D) (x: float32 clarray) (y: float32 clarray) (z: float32 clarray) ->
                let i = range.GlobalID0
                if i < chunkSize then
                    z.[i] <- x.[i] + y.[i]
        @>

    let chunk = n / devices.Length

    devices
    |> Array.iteri (fun idx device ->
        let start = idx * chunk
        let size = if idx = devices.Length - 1 then n - start else chunk

        let subA = a.[start .. start + size - 1]
        let subB = b.[start .. start + size - 1]

        let context = RuntimeContext(device)
        let subC =
            opencl {
                use! dx = ClArray.toDevice subA
                use! dy = ClArray.toDevice subB
                use! dz = ClArray.alloc<float32> size
                do! runCommand (kernel size) <| fun k -> k <| validRange size <| dx <| dy <| dz
                return! ClArray.toHost dz
            }
            |> runOpenCl context

        Array.blit subC 0 output start size)

    let ok = output |> Array.mapi (fun i v -> abs (v - (a.[i] + b.[i])) < 1e-5f) |> Array.forall id
    printfn "Devices used: %d" devices.Length
    devices |> Array.iteri (fun i d -> printfn "  [%d] %s" i d.Name)
    printfn "%s" (if ok then "Multi-device task verified." else "Multi-device task failed.")
