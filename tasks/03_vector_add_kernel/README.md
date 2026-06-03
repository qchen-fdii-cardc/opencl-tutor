# 任务 03: 向量加法内核

## 任务目标
实现第一个可执行 Kernel：`C = A + B`，并与 CPU 结果校验。

## 学习动机
这是 OpenCL 最经典“通路任务”：主机数据 -> 设备计算 -> 主机读回。

## 关键概念
- **Program/Kernel**：运行时编译与内核对象。
- **Buffer**：主机与设备之间的数据容器。
- **NDRange**：全局工作项数量。

## 关键函数与常量
- `clCreateProgramWithSource` / `clBuildProgram`：编译内核。
- `clCreateKernel` / `clSetKernelArg`：准备内核调用。
- `clCreateBuffer`：创建输入输出缓冲区。
- `clEnqueueNDRangeKernel` / `clEnqueueReadBuffer`：执行并读回。
- `CL_MEM_COPY_HOST_PTR` / `CL_MEM_WRITE_ONLY`：内存访问语义。

## 代码实现思路
1. 构造输入向量 `a`、`b`。
2. 编译 `vec_add` 内核。
3. 分配 3 个 Buffer（A/B/C）。
4. 提交 1D NDRange。
5. 读回并逐元素校验。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_03.exe
```

## 验收标准
- 输出 `Vector add verified.`。
- 任意元素误差不超过 `1e-5`。

## 常见错误与排查
- `Build failed`：检查内核源码字符串语法。
- 结果全 0：确认 `clSetKernelArg` 顺序与类型。
- 结果错乱：确认 `global_size == 向量长度`。
