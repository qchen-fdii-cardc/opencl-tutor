# 任务 05: 内核执行计时

## 任务目标
使用 `Event Profiling` 获取内核执行时间（纳秒级）。

## 学习动机
性能优化前先建立可重复测量能力；否则优化结论不可信。

## 关键概念
- **Profiling Queue**：需启用 `CL_QUEUE_PROFILING_ENABLE`。
- **Event 时间戳**：使用 `CL_PROFILING_COMMAND_START/END`。
- **计算与传输分离**：本任务仅统计 Kernel 执行段。

## 关键函数与常量
- `clCreateCommandQueue(..., CL_QUEUE_PROFILING_ENABLE, ...)`
- `clEnqueueNDRangeKernel(..., &event)`
- `clGetEventProfilingInfo`
- `CL_PROFILING_COMMAND_START` / `CL_PROFILING_COMMAND_END`

## 代码实现思路
1. 用 `scale` 内核处理大向量。
2. 队列开启 profiling。
3. 获取 kernel event 并读取开始/结束时间。
4. 转换为毫秒输出。
5. 校验计算结果正确性。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_05.exe
```

## 验收标准
- 输出 `Kernel time (profiling): ... ms`。
- 输出 `Timing task verified.`。

## 常见错误与排查
- profiling 时间为 0：确认队列启用了 `CL_QUEUE_PROFILING_ENABLE`。
- `clGetEventProfilingInfo` 失败：确认事件来自可 profiling 队列。
