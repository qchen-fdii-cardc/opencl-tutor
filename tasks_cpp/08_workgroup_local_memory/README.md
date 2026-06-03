# 任务 08: 工作组与 local memory 优化

## 任务目标
实现基于 `__local` 缓存的并行归约（求和），理解工作组同步。

## 学习动机
`local memory` 是 OpenCL 优化核心手段之一，适合在组内复用数据。

## 关键概念
- **Work-group**：局部同步范围。
- **Local Memory**：组内共享高速缓存。
- **barrier**：保证组内线程同步。

## 关键函数与常量
- 核函数参数中的 `__local float*`
- `barrier(CLK_LOCAL_MEM_FENCE)`
- `get_local_id` / `get_group_id` / `get_local_size`

## 代码实现思路
1. 输入向量分块到多个工作组。
2. 每组将数据写入 `cache`（local memory）。
3. 组内做二分归约。
4. 每组输出一个 partial sum，主机端最终求和。
5. 与 CPU `std::accumulate` 对比。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_08.exe
```

## 验收标准
- 输出 CPU 与 GPU（局部归约）求和结果。
- 输出 `Local memory reduction verified.`。

## 常见错误与排查
- 结果偏小：确认 global size 对齐到 local size。
- 随机错误：确认每轮归约后都调用 barrier。
