# OpenCL 学习路径（CMake 工程版）

本仓库按照“从基础到进阶”的顺序组织 OpenCL 学习任务。每个任务都在独立目录中，包含：
- README.md：中文任务说明、概念与验收标准
- CMakeLists.txt：独立构建入口
- src/main.cpp：示例代码（已实现或占位）

## 学习任务总览

1. [任务 01: 列出平台与设备](tasks/01_platform_device_inventory)
2. [任务 02: 创建上下文与命令队列](tasks/02_context_and_queue)
3. [任务 03: 向量加法内核](tasks/03_vector_add_kernel)
4. [任务 04: 统一错误处理](tasks/04_error_handling)
5. [任务 05: 内核执行计时](tasks/05_kernel_timing)
6. [任务 06: 内存模型与缓冲区](tasks/06_memory_model_buffers)
7. [任务 07: 矩阵乘法内核](tasks/07_matrix_multiply)
8. [任务 08: 工作组与局部内存优化](tasks/08_workgroup_local_memory)
9. [任务 09: CPU 与 OpenCL 性能对比](tasks/09_cpu_vs_opencl_benchmark)
10. [任务 10: 图像与采样器基础](tasks/10_images_and_samplers)
11. [任务 11: 高斯模糊图像处理](tasks/11_gaussian_blur)
12. [任务 12: 设备扩展探索](tasks/12_extensions)
13. [任务 13: 非图形场景实践](tasks/13_nongraphics_apps)
14. [任务 14: 多设备并行](tasks/14_multi_device_parallel)

## 当前进度

- 已完成：任务 01（平台与设备枚举）
- 进行中：后续任务按目录逐步实现

## 构建方式

每个任务独立构建，例如任务 01：

```powershell
cmake -S tasks/01_platform_device_inventory -B build/task01
cmake --build build/task01 --config Debug
.\\build\\task01\\Debug\\task_01.exe
```

## 推荐参考资料

- OpenCL 官方规范（Khronos）
- OpenCL-Guide（KhronosGroup）
- 设备厂商开发文档（NVIDIA / AMD / Intel）

## 版本控制建议

建议每完成一个任务就提交一次，提交信息示例：
- feat(task03): add vector addition kernel
- docs(task06): explain OpenCL memory flags

