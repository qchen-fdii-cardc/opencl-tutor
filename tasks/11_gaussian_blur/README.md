# 任务 11: 高斯模糊（OpenCL 与 CPU 对比）

## 任务目标
实现 3x3 高斯模糊核，并对比 CPU 与 OpenCL 的结果与耗时。

## 学习动机
这是典型图像计算任务，可同时锻炼边界处理、二维 NDRange 与性能分析。

## 关键概念
- **卷积核**：`[1 2 1; 2 4 2; 1 2 1] / 16`。
- **边界处理**：通过 clamp 限制坐标越界。
- **双通道验证**：先验证结果，再比较性能。

## 关键函数与常量
- `clEnqueueNDRangeKernel`（2D）
- `clGetEventProfilingInfo`
- Kernel 内置函数 `clamp`

## 代码实现思路
1. 生成灰度测试图像。
2. CPU 实现 3x3 高斯模糊并计时。
3. OpenCL 实现同等算法并计时。
4. 逐像素比较误差（`1e-4`）。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_11.exe
```

## 验收标准
- 输出 CPU 与 OpenCL 模糊耗时。
- 输出 `Gaussian blur comparison verified.`。

## 常见错误与排查
- 结果不一致：检查 CPU 与 Kernel 权重和边界逻辑是否一致。
- OpenCL 时间为 0：确认使用 profiling 队列。
