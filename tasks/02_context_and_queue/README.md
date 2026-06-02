# 任务 02: 创建上下文与命令队列

## 任务目标
在可用设备上创建 `cl_context` 与 `cl_command_queue`，并正确释放资源。

## 学习动机
后续所有 Buffer、Program、Kernel、Event 都依赖 Context/Queue。本任务是“能提交命令”的起点。

## 关键概念
- **Context**：OpenCL 资源容器。
- **Command Queue**：向设备提交命令的队列。
- **生命周期管理**：`clCreate*` 与 `clRelease*` 必须成对出现。

## 关键函数与常量
- `clGetPlatformIDs` / `clGetDeviceIDs`：选择设备。
- `clCreateContext`：创建上下文。
- `clCreateCommandQueue`：创建命令队列（本仓库保持 OpenCL 1.2 风格）。
- `CL_DEVICE_TYPE_ALL`：枚举所有设备类型。
- `CL_SUCCESS`：调用成功返回码。

## 代码实现思路
1. 枚举平台和设备，选第一块可用设备。
2. 创建 Context。
3. 创建 Command Queue。
4. 输出成功信息。
5. 释放 Queue 和 Context。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_02.exe
```

## 验收标准
- 能输出 `Context and command queue created successfully.`。
- 没有资源泄漏（创建的 OpenCL 对象均释放）。

## 常见错误与排查
- `No usable OpenCL device found`：检查驱动与运行时。
- `clCreateContext failed`：设备句柄无效或运行时异常。
- `clCreateCommandQueue failed`：设备不支持或上下文不匹配。
