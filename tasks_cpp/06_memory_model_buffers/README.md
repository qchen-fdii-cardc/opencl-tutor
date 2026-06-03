# 任务 06: 内存模型与缓冲区

## 任务目标
通过 SAXPY 示例演示 `CL_MEM_USE_HOST_PTR`、`CL_MEM_COPY_HOST_PTR`、`CL_MEM_ALLOC_HOST_PTR` 的差异与用法。

## 学习动机
OpenCL 性能很大程度取决于数据搬运策略，理解 Buffer 标志是必须掌握的基础。

## 关键概念
- **USE_HOST_PTR**：复用主机地址，适合稳定内存。
- **COPY_HOST_PTR**：创建时复制一份设备侧初值。
- **ALLOC_HOST_PTR**：为映射访问分配适配内存。
- **Map/Unmap**：通过 `clEnqueueMapBuffer` 访问设备缓冲区。

## 关键函数与常量
- `clCreateBuffer`
- `clEnqueueMapBuffer` / `clEnqueueUnmapMemObject`
- `CL_MAP_READ`
- `CL_MEM_USE_HOST_PTR` / `CL_MEM_COPY_HOST_PTR` / `CL_MEM_ALLOC_HOST_PTR`

## 代码实现思路
1. 输入向量 `x`、`y`，执行 `z = a*x + y`。
2. `x` 用 USE_HOST_PTR，`y` 用 COPY_HOST_PTR，`z` 用 ALLOC_HOST_PTR。
3. Kernel 执行后 map `z` 读取结果。
4. 与 CPU 公式对比校验。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_06.exe
```

## 验收标准
- 输出三类内存标志说明。
- 输出 `Memory task verified.`。

## 常见错误与排查
- map 返回空指针：检查 `clEnqueueMapBuffer` 返回码。
- 数值不一致：检查 buffer 创建标志与参数顺序。
