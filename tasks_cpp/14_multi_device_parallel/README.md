# 任务 14: 多设备并行计算

## 任务目标

把向量加法任务切分到多设备执行（最多使用前 2 个设备），并合并结果校验。

## 学习动机

真实生产任务常需要跨设备扩展吞吐。本任务演示最小可行的“任务分片 + 多设备执行”模式。

## 关键概念

- **任务切片**：将全量输入按区间切分。
- **设备独立上下文**：每个设备单独创建 Context/Queue/Program。
- **结果合并**：按切片偏移写回总结果。

## 关键函数与常量

- `platform.getDevices(...)`：枚举设备池。
- `cl::Context` / `cl::CommandQueue`：每设备资源创建。
- `queue.enqueueNDRangeKernel(...)`：每设备执行对应 chunk。

## 代码实现思路

1. 枚举设备并取前 2 个（若不足则降级）。
2. 将输入向量按设备数切块。
3. 每块在对应设备上执行 `vec_add`。
4. 合并所有 chunk，和 CPU 参考结果比对。

## 构建与运行

```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_14.exe
```

## 验收标准

- 输出使用设备数量。
- 输出 `Multi-device task verified.`。

## 常见错误与排查

- 仅 1 个设备：程序会自动退化为单设备路径。
- 结果错位：检查 chunk 偏移与写回下标。
