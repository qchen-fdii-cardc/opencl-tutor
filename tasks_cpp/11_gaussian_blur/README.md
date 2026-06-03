# 任务 11: 高斯模糊（OpenCL 与 CPU 对比）

## 任务目标

实现 3x3 高斯模糊核并验证输出发生平滑变化。

## 学习动机

这是典型图像计算任务，可锻炼边界处理与二维 NDRange 映射。

## 关键概念

- **卷积核**：`[1 2 1; 2 4 2; 1 2 1] / 16`。
- **边界处理**：通过 clamp 限制坐标越界。
- **双通道验证**：先验证结果，再比较性能。

## 关键函数与常量

- `queue.enqueueNDRangeKernel`（2D）
- Kernel 内置函数 `clamp`

## 代码实现思路

1. 生成灰度测试图像。
2. OpenCL 实现 3x3 高斯模糊。
3. 读回结果并确认图像相对输入发生变化。

## 构建与运行

```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_11.exe
```

## 验收标准

- 输出 `Gaussian blur task verified.`。

## 常见错误与排查

- 结果不一致：检查 CPU 与 Kernel 权重和边界逻辑是否一致。
- 结果几乎不变：检查 kernel 是否真正写入输出图像。
