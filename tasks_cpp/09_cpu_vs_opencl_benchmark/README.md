# 任务 09: CPU vs OpenCL 性能对比

## 任务目标

在同一向量加法问题上比较 CPU 与 OpenCL 内核耗时，并验证结果一致性。

## 学习动机

学习 OpenCL 不仅要“能跑”，还要形成“性能对比 + 正确性校验”的实验习惯。

## 关键概念

- **基线测试**：CPU 实现作为对照组。
- **Kernel Profiling**：OpenCL 内核计时。
- **Speedup**：`CPU_time / OpenCL_time`。

## 关键函数与常量

- C++ `std::chrono`（CPU 计时）
- `event.getProfilingInfo<...>()`（OpenCL 计时）
- `CL_QUEUE_PROFILING_ENABLE`

## 代码实现思路

1. 生成大向量输入。
2. CPU 顺序加法并计时。
3. OpenCL `vec_add` 内核并计时。
4. 校验 CPU/GPU 结果逐元素一致。
5. 打印 speedup。

## 构建与运行

```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_09.exe
```

## 验收标准

- 输出 CPU 时间、OpenCL 时间、加速比。
- 输出 `Benchmark result verified.`。

## 常见错误与排查

- speedup 异常：确认比较的是同等工作量。
- 数据不一致：检查读回长度和类型。
