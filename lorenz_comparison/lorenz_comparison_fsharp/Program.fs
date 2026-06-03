open System
open System.Diagnostics
open System.Globalization
open System.IO
open Brahma.FSharp

type LorenzParams =
    { Sigma: float
      Rho: float
      Beta: float
      Dt: float
      Steps: int
      Trajectories: int }

type BenchmarkResult =
    { ElapsedSeconds: float
      X: float[]
      Y: float[]
      Z: float[] }

type HistoryResult = { X: float[]; Y: float[]; Z: float[] }

let defaultParams =
    { Sigma = 10.0
      Rho = 28.0
      Beta = 8.0 / 3.0
      Dt = 1e-3
      Steps = 20000
      Trajectories = 8192 }

let parseArgs (argv: string[]) =
    let mutable p = defaultParams

    if argv.Length > 0 then
        p <- { p with Steps = max 1 (int argv.[0]) }

    if argv.Length > 1 then
        p <-
            { p with
                Trajectories = max 1 (int argv.[1]) }

    p

let initializeTrajectories n =
    let x = Array.zeroCreate<float> n
    let y = Array.zeroCreate<float> n
    let z = Array.zeroCreate<float> n

    for i in 0 .. n - 1 do
        let f = float i
        x.[i] <- -8.0 + 0.0001 * f
        y.[i] <- 8.0 - 0.0001 * f
        z.[i] <- 27.0 + 0.00005 * f

    x, y, z

let rk4Step (p: LorenzParams) (x: float) (y: float) (z: float) =
    let dx1 = p.Sigma * (y - x)
    let dy1 = x * (p.Rho - z) - y
    let dz1 = x * y - p.Beta * z

    let x2 = x + 0.5 * p.Dt * dx1
    let y2 = y + 0.5 * p.Dt * dy1
    let z2 = z + 0.5 * p.Dt * dz1
    let dx2 = p.Sigma * (y2 - x2)
    let dy2 = x2 * (p.Rho - z2) - y2
    let dz2 = x2 * y2 - p.Beta * z2

    let x3 = x + 0.5 * p.Dt * dx2
    let y3 = y + 0.5 * p.Dt * dy2
    let z3 = z + 0.5 * p.Dt * dz2
    let dx3 = p.Sigma * (y3 - x3)
    let dy3 = x3 * (p.Rho - z3) - y3
    let dz3 = x3 * y3 - p.Beta * z3

    let x4 = x + p.Dt * dx3
    let y4 = y + p.Dt * dy3
    let z4 = z + p.Dt * dz3
    let dx4 = p.Sigma * (y4 - x4)
    let dy4 = x4 * (p.Rho - z4) - y4
    let dz4 = x4 * y4 - p.Beta * z4

    let xn = x + (p.Dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
    let yn = y + (p.Dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
    let zn = z + (p.Dt / 6.0) * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4)
    xn, yn, zn

let runCpuBenchmark (p: LorenzParams) (initX: float[]) (initY: float[]) (initZ: float[]) =
    let x = Array.copy initX
    let y = Array.copy initY
    let z = Array.copy initZ

    let sw = Stopwatch.StartNew()

    for i in 0 .. p.Trajectories - 1 do
        let mutable xv = x.[i]
        let mutable yv = y.[i]
        let mutable zv = z.[i]

        for _step in 1 .. p.Steps do
            let xn, yn, zn = rk4Step p xv yv zv
            xv <- xn
            yv <- yn
            zv <- zn

        x.[i] <- xv
        y.[i] <- yv
        z.[i] <- zv

    sw.Stop()

    { ElapsedSeconds = sw.Elapsed.TotalSeconds
      X = x
      Y = y
      Z = z }

let runCpuHistory (p: LorenzParams) (x0: float) (y0: float) (z0: float) =
    let xh = Array.zeroCreate<float> (p.Steps + 1)
    let yh = Array.zeroCreate<float> (p.Steps + 1)
    let zh = Array.zeroCreate<float> (p.Steps + 1)

    let mutable x = x0
    let mutable y = y0
    let mutable z = z0

    xh.[0] <- x
    yh.[0] <- y
    zh.[0] <- z

    for i in 1 .. p.Steps do
        let xn, yn, zn = rk4Step p x y z
        x <- xn
        y <- yn
        z <- zn
        xh.[i] <- x
        yh.[i] <- y
        zh.[i] <- z

    { X = xh; Y = yh; Z = zh }

let chooseDevice () =
    let devices =
        ClDevice.GetAvailableDevices(Platform.Any, DeviceType.Default) |> Seq.toArray

    if devices.Length = 0 then
        failwith "No OpenCL device found."

    let tryByName (term: string) =
        devices
        |> Array.tryFind (fun d -> d.Name.IndexOf(term, StringComparison.OrdinalIgnoreCase) >= 0)

    match tryByName "NVIDIA" with
    | Some d -> d
    | None ->
        printfn "[WARN] NVIDIA device not found. Falling back to: %s" devices.[0].Name
        devices.[0]

let runOpenClBenchmark (context: RuntimeContext) (p: LorenzParams) (initX: float[]) (initY: float[]) (initZ: float[]) =
    let n = p.Trajectories
    let range = Range1D.CreateValid(n, min 256 n)
    let dt = p.Dt
    let sigma = p.Sigma
    let rho = p.Rho
    let beta = p.Beta

    let kernel =
        <@
            fun
                (range: Range1D)
                (inX: float clarray)
                (inY: float clarray)
                (inZ: float clarray)
                (outX: float clarray)
                (outY: float clarray)
                (outZ: float clarray)
                count
                steps ->
                let i = range.GlobalID0

                if i < count then
                    let mutable x = inX.[i]
                    let mutable y = inY.[i]
                    let mutable z = inZ.[i]

                    for _s in 1..steps do
                        let dx1 = sigma * (y - x)
                        let dy1 = x * (rho - z) - y
                        let dz1 = x * y - beta * z

                        let x2 = x + 0.5 * dt * dx1
                        let y2 = y + 0.5 * dt * dy1
                        let z2 = z + 0.5 * dt * dz1
                        let dx2 = sigma * (y2 - x2)
                        let dy2 = x2 * (rho - z2) - y2
                        let dz2 = x2 * y2 - beta * z2

                        let x3 = x + 0.5 * dt * dx2
                        let y3 = y + 0.5 * dt * dy2
                        let z3 = z + 0.5 * dt * dz2
                        let dx3 = sigma * (y3 - x3)
                        let dy3 = x3 * (rho - z3) - y3
                        let dz3 = x3 * y3 - beta * z3

                        let x4 = x + dt * dx3
                        let y4 = y + dt * dy3
                        let z4 = z + dt * dz3
                        let dx4 = sigma * (y4 - x4)
                        let dy4 = x4 * (rho - z4) - y4
                        let dz4 = x4 * y4 - beta * z4

                        x <- x + (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
                        y <- y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
                        z <- z + (dt / 6.0) * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4)

                    outX.[i] <- x
                    outY.[i] <- y
                    outZ.[i] <- z
        @>

    let sw = Stopwatch.StartNew()

    let gx, gy, gz =
        opencl {
            use! inX = ClArray.toDevice initX
            use! inY = ClArray.toDevice initY
            use! inZ = ClArray.toDevice initZ
            use! outX = ClArray.alloc<float> n
            use! outY = ClArray.alloc<float> n
            use! outZ = ClArray.alloc<float> n

            do!
                runCommand kernel
                <| fun k -> k <| range <| inX <| inY <| inZ <| outX <| outY <| outZ <| n <| p.Steps

            let! rx = ClArray.toHost outX
            let! ry = ClArray.toHost outY
            let! rz = ClArray.toHost outZ
            return rx, ry, rz
        }
        |> ClTask.runSync context

    sw.Stop()

    { ElapsedSeconds = sw.Elapsed.TotalSeconds
      X = gx
      Y = gy
      Z = gz }

let runOpenClHistory (context: RuntimeContext) (p: LorenzParams) (x0: float) (y0: float) (z0: float) =
    let dt = p.Dt
    let sigma = p.Sigma
    let rho = p.Rho
    let beta = p.Beta

    let kernel =
        <@
            fun (_range: Range1D) (xHist: float clarray) (yHist: float clarray) (zHist: float clarray) steps ->
                let mutable x = x0
                let mutable y = y0
                let mutable z = z0

                xHist.[0] <- x
                yHist.[0] <- y
                zHist.[0] <- z

                for i in 1..steps do
                    let dx1 = sigma * (y - x)
                    let dy1 = x * (rho - z) - y
                    let dz1 = x * y - beta * z

                    let x2 = x + 0.5 * dt * dx1
                    let y2 = y + 0.5 * dt * dy1
                    let z2 = z + 0.5 * dt * dz1
                    let dx2 = sigma * (y2 - x2)
                    let dy2 = x2 * (rho - z2) - y2
                    let dz2 = x2 * y2 - beta * z2

                    let x3 = x + 0.5 * dt * dx2
                    let y3 = y + 0.5 * dt * dy2
                    let z3 = z + 0.5 * dt * dz2
                    let dx3 = sigma * (y3 - x3)
                    let dy3 = x3 * (rho - z3) - y3
                    let dz3 = x3 * y3 - beta * z3

                    let x4 = x + dt * dx3
                    let y4 = y + dt * dy3
                    let z4 = z + dt * dz3
                    let dx4 = sigma * (y4 - x4)
                    let dy4 = x4 * (rho - z4) - y4
                    let dz4 = x4 * y4 - beta * z4

                    x <- x + (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
                    y <- y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
                    z <- z + (dt / 6.0) * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4)

                    xHist.[i] <- x
                    yHist.[i] <- y
                    zHist.[i] <- z
        @>

    let n = p.Steps + 1
    let range = Range1D.CreateValid(1, 1)

    let x, y, z =
        opencl {
            use! xHist = ClArray.alloc<float> n
            use! yHist = ClArray.alloc<float> n
            use! zHist = ClArray.alloc<float> n

            do! runCommand kernel <| fun k -> k <| range <| xHist <| yHist <| zHist <| p.Steps

            let! rx = ClArray.toHost xHist
            let! ry = ClArray.toHost yHist
            let! rz = ClArray.toHost zHist
            return rx, ry, rz
        }
        |> ClTask.runSync context

    { X = x; Y = y; Z = z }

let computeMaxDiff3 (ax: float[]) (ay: float[]) (az: float[]) (bx: float[]) (by: float[]) (bz: float[]) =
    let mutable maxDiff = 0.0

    for i in 0 .. ax.Length - 1 do
        let dx = abs (ax.[i] - bx.[i])
        let dy = abs (ay.[i] - by.[i])
        let dz = abs (az.[i] - bz.[i])
        maxDiff <- max maxDiff (max dx (max dy dz))

    maxDiff

let writeValidationCsv (path: string) (p: LorenzParams) (cpu: HistoryResult) (gpu: HistoryResult) =
    let culture = CultureInfo.InvariantCulture
    use writer = new StreamWriter(path, false)

    writer.WriteLine("step,time,cpu_x,cpu_y,cpu_z,gpu_x,gpu_y,gpu_z,abs_dx,abs_dy,abs_dz,max_abs_diff")

    for i in 0 .. p.Steps do
        let time = p.Dt * float i
        let dx = abs (cpu.X.[i] - gpu.X.[i])
        let dy = abs (cpu.Y.[i] - gpu.Y.[i])
        let dz = abs (cpu.Z.[i] - gpu.Z.[i])
        let maxDiff = max dx (max dy dz)

        writer.Write(string i)
        writer.Write(",")
        writer.Write(time.ToString("R", culture))
        writer.Write(",")
        writer.Write(cpu.X.[i].ToString("R", culture))
        writer.Write(",")
        writer.Write(cpu.Y.[i].ToString("R", culture))
        writer.Write(",")
        writer.Write(cpu.Z.[i].ToString("R", culture))
        writer.Write(",")
        writer.Write(gpu.X.[i].ToString("R", culture))
        writer.Write(",")
        writer.Write(gpu.Y.[i].ToString("R", culture))
        writer.Write(",")
        writer.Write(gpu.Z.[i].ToString("R", culture))
        writer.Write(",")
        writer.Write(dx.ToString("R", culture))
        writer.Write(",")
        writer.Write(dy.ToString("R", culture))
        writer.Write(",")
        writer.Write(dz.ToString("R", culture))
        writer.Write(",")
        writer.WriteLine(maxDiff.ToString("R", culture))

[<EntryPoint>]
let main argv =
    try
        let p = parseArgs argv
        let initX, initY, initZ = initializeTrajectories p.Trajectories

        let device = chooseDevice ()
        let context = RuntimeContext(device)

        printfn "OpenCL device selected: %s | Platform: %A" device.Name device.Platform

        let cpuBenchmark = runCpuBenchmark p initX initY initZ
        let gpuBenchmark = runOpenClBenchmark context p initX initY initZ

        let benchmarkMaxDiff =
            computeMaxDiff3 cpuBenchmark.X cpuBenchmark.Y cpuBenchmark.Z gpuBenchmark.X gpuBenchmark.Y gpuBenchmark.Z

        let x0, y0, z0 = -8.0, 8.0, 27.0
        let cpuHistory = runCpuHistory p x0 y0 z0
        let gpuHistory = runOpenClHistory context p x0 y0 z0

        let historyMaxDiff =
            computeMaxDiff3 cpuHistory.X cpuHistory.Y cpuHistory.Z gpuHistory.X gpuHistory.Y gpuHistory.Z

        let csvPath =
            Path.Combine("lorenz_comparison", "lorenz_comparison_fsharp", "lorenz_validation_fsharp.csv")

        let outDir = Path.GetDirectoryName(csvPath)

        if not (String.IsNullOrWhiteSpace(outDir)) then
            Directory.CreateDirectory(outDir) |> ignore

        writeValidationCsv csvPath p cpuHistory gpuHistory

        printfn "CPU benchmark time (s): %.6f" cpuBenchmark.ElapsedSeconds
        printfn "OpenCL benchmark time (s): %.6f" gpuBenchmark.ElapsedSeconds
        printfn "Speedup (CPU/OpenCL): %.6fx" (cpuBenchmark.ElapsedSeconds / max gpuBenchmark.ElapsedSeconds 1e-12)
        printfn "Max abs diff (benchmark final states): %.12e" benchmarkMaxDiff
        printfn "Max abs diff (history CSV): %.12e" historyMaxDiff
        printfn "Validation CSV written: %s" csvPath
        printfn "Args: [steps] [trajectories], current=%d, %d" p.Steps p.Trajectories
        0
    with ex ->
        eprintfn "Error: %s" ex.Message
        1
