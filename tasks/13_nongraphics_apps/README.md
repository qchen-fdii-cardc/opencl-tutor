# 任务 13: 非图形场景实践

## 任务目标
用 OpenCL 实现 Monte Carlo 估算圆周率（π），展示非图形计算场景。

## 学习动机
OpenCL 不只用于图像；金融仿真、科学计算、统计采样都属于典型非图形工作负载。

## 关键概念
- **Monte Carlo**：大量随机采样近似求解。
- **数据并行**：每个 work-item 独立计算命中次数。
- **主机归约**：设备输出局部结果，主机做最终汇总。

## 关键函数与常量
- `clEnqueueNDRangeKernel`
- `clEnqueueReadBuffer`
- Kernel 侧随机数推进（LCG）

## 代码实现思路
1. 为每个 work-item 准备独立随机种子。
2. Kernel 执行 `samples_per_item` 次采样，统计单位圆命中数。
3. 主机汇总总命中率并估算 π。
4. 与真实 π 比较绝对误差。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_13.exe
```

## 验收标准
- 输出估算值与误差。
- 输出 `Non-graphics OpenCL task verified.`（误差阈值 `< 0.05`）。

## 常见错误与排查
- 误差过大：增加样本量（work-items 或 samples_per_item）。
- 结果不稳定：检查随机种子是否分散。
