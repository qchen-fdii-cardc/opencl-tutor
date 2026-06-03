# 任务 07: 矩阵乘法与正确性验证

## 任务目标
实现 `N×N` 矩阵乘法内核，并与 CPU 参考实现逐元素对比。

## 学习动机
矩阵乘法是很多高性能计算任务的基础，也是理解二维 NDRange 的经典案例。

## 关键概念
- **二维索引映射**：`row = get_global_id(1)`，`col = get_global_id(0)`。
- **全局内存访问模式**：按行列展开的一维数组。
- **正确性优先**：先保证结果一致，再做优化。

## 关键函数与常量
- `queue.enqueueNDRangeKernel`（2D）
- `get_global_id(0/1)`
- `CL_MEM_READ_ONLY` / `CL_MEM_WRITE_ONLY`

## 代码实现思路
1. 构造 `a`、`b` 两个输入矩阵。
2. CPU 端计算 `c_cpu` 作为金标准。
3. OpenCL 执行 `mat_mul` 得到 `c_gpu`。
4. 用误差阈值 `1e-3` 做逐元素验证。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_07.exe
```

## 验收标准
- 输出 `Matrix multiply verified.`。
- GPU 与 CPU 结果一致（误差阈值内）。

## 常见错误与排查
- 结果全错：检查 row/col 映射是否交换。
- 越界访问：核函数中保留边界判断。
