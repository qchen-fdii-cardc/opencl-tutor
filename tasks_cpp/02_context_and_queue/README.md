# 任务 02: 创建上下文与命令队列

## 任务目标

在可用设备上创建 `cl::Context` 与 `cl::CommandQueue`，并依赖 RAII 自动管理资源。

## 学习动机

后续所有 Buffer、Program、Kernel、Event 都依赖 Context/Queue。本任务是“能提交命令”的起点。

## 关键概念

- **Context**：OpenCL 资源容器。
- **Command Queue**：向设备提交命令的队列。
- **生命周期管理**：`cl::` 对象离开作用域自动释放（RAII）。

## 关键函数与常量

- `cl::Platform::get` / `platform.getDevices(...)`：选择设备。
- `cl::Context`：创建上下文。
- `cl::CommandQueue`：创建命令队列。
- `CL_DEVICE_TYPE_ALL`：枚举所有设备类型。
- `cl::Error`：异常模式下的错误报告。

## 代码实现思路

1. 枚举平台和设备，选第一块可用设备。
2. 创建 Context。
3. 创建 Command Queue。
4. 输出成功信息。
5. 离开作用域后自动释放 Queue 和 Context。

## 构建与运行

```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_02.exe
```

## 验收标准

- 能输出 `Context and command queue created successfully.`。
- 没有资源泄漏（由 RAII 自动回收）。

## 常见错误与排查

- `No usable OpenCL device found`：检查驱动与运行时。
- `OpenCL error: ...`：查看异常中的错误码与调用阶段。
