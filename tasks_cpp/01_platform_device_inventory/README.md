# 任务 01: 列出平台与设备

本任务是 OpenCL 学习路径的第一步，目标不是“跑出 GPU 结果”，而是先建立一个稳定的观察窗口，确认系统里到底有哪些 OpenCL 平台和设备可用。

## 任务目标

枚举系统中的 OpenCL 平台和设备，打印平台名、设备名与设备类型。通过这个任务，你可以先确认驱动是否安装正确、设备是否被 OpenCL 识别，以及后续任务应该选用哪一块设备。

## 你会学到什么

- OpenCL 的平台模型：一个系统可以暴露多个平台，每个平台通常代表一个厂商实现。
- OpenCL 的设备模型：平台下面可以挂载 CPU、GPU、加速器等不同设备。
- 如何用 `cl::Platform::get` 和 `platform.getDevices(...)` 进行枚举。
- 如何用 `platform.getInfo<...>()` 和 `device.getInfo<...>()` 读取字符串信息和设备属性。
- 如何处理“没有平台”“没有设备”“查询失败”等基础错误场景。

## 概念说明

### Platform

Platform 可以理解为 OpenCL 运行时的“入口”。它把同一套厂商实现的设备组织在一起。一个系统上可能同时存在多个平台，例如不同 GPU 厂商或 CPU OpenCL 运行时。

### Device

Device 是真正执行 OpenCL 计算的硬件或虚拟计算单元。常见类型包括：

- `CL_DEVICE_TYPE_CPU`：CPU 设备。
- `CL_DEVICE_TYPE_GPU`：GPU 设备。
- `CL_DEVICE_TYPE_ACCELERATOR`：专用加速器。
- `CL_DEVICE_TYPE_CUSTOM`：厂商自定义设备。

### Enumerate

“枚举”就是先问系统“有多少个平台”，再取出平台列表；然后对每个平台继续问“有多少设备”，再取出设备列表。这个模式在 OpenCL 中非常常见。

## 本任务使用的 C++ API

### `cl::Platform::get`

作用：获取系统中的 OpenCL 平台对象列表。

典型调用方式：

```cpp
std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms);
```

参数说明：

- `platforms` 会被填充为当前系统可见的平台对象列表。
- 平台数量可通过 `platforms.size()` 获取。

返回值：

- 成功时列表非空；失败通常通过异常体现（启用异常时）。

### `platform.getInfo<CL_PLATFORM_NAME>()`

作用：查询平台信息，例如平台名称。

本任务中使用了 `CL_PLATFORM_NAME` 来打印平台名。调用流程是先查询字符串长度，再分配缓冲区，最后再次读取字符串。

### `platform.getDevices(...)`

作用：查询某个平台下有哪些设备。

本任务中使用了：

- `CL_DEVICE_TYPE_ALL`：列出所有类型的设备。
- 通过 `platform.getDevices(CL_DEVICE_TYPE_ALL, &devices)` 一次获取设备对象列表。

### `device.getInfo<...>()`

作用：查询设备属性。

本任务使用了：

- `CL_DEVICE_NAME`：设备名称。
- `CL_DEVICE_TYPE`：设备类型，用于判断是 CPU、GPU 还是其他类型。

## 本任务使用的常量

### `CL_DEVICE_TYPE_ALL`

表示查询所有设备类型，而不是只看 GPU 或 CPU。这样更适合教学场景，因为可以先完整看见系统暴露了什么。

### `CL_DEVICE_NAME`

设备名称属性，通常会返回类似 `NVIDIA GeForce RTX 3070` 这样的字符串。

### `CL_DEVICE_TYPE`

设备类型属性，返回的是一个位掩码。代码里用它来判断设备属于哪个大类。

### `CL_PLATFORM_NAME`

平台名称属性，通常会显示平台实现的厂商名称或运行时名称。

### `CL_SUCCESS`

表示 OpenCL 调用成功的标准返回码。只要不是 `CL_SUCCESS`，就说明调用失败，需要进一步处理。

## 代码实现说明

### 1. 先查询平台数量

程序先调用 `cl::Platform::get(&platforms)`，再通过 `platforms.size()` 得到平台数量。

### 2. 再读取平台列表

平台对象会直接存放在 `std::vector<cl::Platform>` 中，不需要手动管理底层 C 句柄数组。

### 3. 打印平台名称

平台名可通过 `platform.getInfo<CL_PLATFORM_NAME>()` 直接获取。

### 4. 查询每个平台下的设备

对每个平台调用 `platform.getDevices(CL_DEVICE_TYPE_ALL, &devices)` 获取设备列表。

### 5. 打印设备名称和设备类型

`device.getInfo<CL_DEVICE_NAME>()` 读取设备名，`device.getInfo<CL_DEVICE_TYPE>()` 的结果可翻译成更易读文本。

### 6. 处理失败和空结果

如果平台数量为 0，程序会直接提示 `No OpenCL platform found.`。若查询设备失败或某个平台没有设备，程序会跳过该平台并继续处理其他平台。

## 关键函数逐个说明

### `device_type_to_string(cl_device_type type)`

这个函数把设备类型常量转换成字符串，方便输出阅读。

- 输入：`cl_device_type`，是一个位掩码。
- 输出：`GPU`、`CPU`、`ACCELERATOR`、`CUSTOM` 或 `DEFAULT/UNKNOWN`。

这里的判断顺序是有意义的，因为一个设备类型值可能包含某些位标志。当前实现以最常见的类型优先匹配。

### `get_platform_name(const cl::Platform& platform)`

这个函数负责读取平台名称。

实现步骤：

1. 先查询名称字符串长度。
2. 创建足够大的字符缓冲区。
3. 再次读取平台名。
4. 把 `char` 缓冲区转换成 `std::string`。

在 C++ API 中，也可以直接使用 `platform.getInfo<CL_PLATFORM_NAME>()` 获得字符串结果。

### `get_device_name(const cl::Device& device)`

这个函数与 `get_platform_name` 类似，只是读取的是设备名而不是平台名。

### `main()`

`main` 函数的职责是控制枚举流程：

1. 查询平台总数。
2. 获取平台列表。
3. 遍历每个平台。
4. 查询该平台上的设备总数。
5. 获取设备列表。
6. 打印设备名称和类型。

## 为什么这样写

### 先查数量，再分配容器

在 C API 里常见“先问数量再写数组”；在本任务使用的 C++ API 里，`std::vector` + `get(...)`/`getDevices(...)` 已经封装了这一步。

### 使用 `std::vector`

这里使用 `std::vector` 管理平台和设备句柄数组，原因是它比手动 `new[]` 更安全，也更符合现代 C++ 风格。

### 只打印，不创建上下文

这一步的目标是确认设备存在并理解 OpenCL 架构，而不是马上进入计算。后续任务会在此基础上创建上下文、命令队列和内核。

## 运行结果应该如何理解

如果你看到类似输出：

```text
OpenCL platform count: 1

[Platform 0] NVIDIA CUDA
 - Device 0: NVIDIA GeForce RTX 3070 [GPU]
```

说明：

- 系统检测到了 1 个 OpenCL 平台。
- 平台名是 `NVIDIA CUDA`。
- 平台下有一块 GPU 设备。

如果没有平台，程序会输出：

```text
No OpenCL platform found.
```

这通常意味着 OpenCL 驱动、运行时或环境没有正确安装。

## 常见问题

### 1. `cl::Platform::get` 失败

可能原因：

- OpenCL 运行时未安装。
- 驱动异常。
- 当前环境没有可见的平台实现。

### 2. 找到了平台，但没有设备

可能原因：

- 该平台没有可用设备。
- 设备驱动未正确加载。
- 当前机器的 OpenCL 环境只暴露了部分能力。

### 3. 输出的设备类型和预期不一致

有些平台会把 CPU、GPU、集成显卡或虚拟设备都暴露出来。枚举结果取决于驱动和平台实现，不同机器之间不完全一致。

## 构建与运行

在任务目录内执行：

```powershell
cmake -S . -B build
cmake --build build --config Debug
.\\build\\Debug\\task_01.exe
```

如果你从仓库根目录构建，也可以使用：

```powershell
cmake -S tasks/01_platform_device_inventory -B build/task01
cmake --build build/task01 --config Debug
.\\build\\task01\\Debug\\task_01.exe
```

## 验收标准

程序正常打印至少一个平台信息；如果无平台，给出清晰提示。

## 下一步

完成这个任务后，下一步就是“创建上下文与命令队列”。那时你会开始真正持有 OpenCL 资源，并为后续 kernel 执行做准备。
