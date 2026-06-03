# OpenCL 基础与关键概念

## 1. 平台（Platform）与设备（Device）
- **Platform**：OpenCL 实现提供方（如 NVIDIA、AMD、Intel）。
- **Device**：实际执行计算的硬件（CPU/GPU/加速器）。
- 一般流程：先枚举平台，再在平台下枚举设备。

## 2. 上下文（Context）
- Context 是 OpenCL 资源容器，绑定一组设备。
- 内存对象（Buffer/Image）、Program、Kernel 等都在 Context 下创建。

## 3. 命令队列（Command Queue）
- 主机通过队列向设备提交命令（数据传输、内核执行、同步）。
- 队列可以是顺序队列或支持乱序执行（取决于设备能力和创建参数）。

## 4. 程序（Program）与内核（Kernel）
- **Program**：内核源码或二进制的容器，需要先编译/构建。
- **Kernel**：Program 中可被调度执行的函数入口。
- 常见流程：`create program -> build -> create kernel -> set args -> enqueue`.

## 5. 内存模型
- **Global Memory**：容量大，延迟高，所有工作项可访问。
- **Local Memory**：工作组共享，适合缓存与协作。
- **Private Memory**：工作项私有变量。
- **Constant Memory**：只读常量区域。

## 6. Buffer 与 Image
- **Buffer**：线性内存，适合向量/矩阵等通用数据。
- **Image**：带格式与坐标语义，适合图像采样与滤波。
- Image 常与 sampler 配合，支持寻址模式和插值模式。

## 7. NDRange、工作项与工作组
- **Work-item**：一次内核执行实例。
- **Work-group**：一组 work-item，能共享 local memory 并做组内同步。
- **Global size** 决定总并行度，**Local size** 决定每组规模。

## 8. 同步与事件（Event）
- OpenCL 操作通常是异步提交。
- 可用 `clFinish`/`queue.finish()` 强制等待，也可通过 Event 建立依赖链。
- 使用 profiling event 可以测量内核和数据传输耗时。

## 9. 主机与设备数据传输
- 常见方式：写入设备（H2D）-> 执行内核 -> 读回主机（D2H）。
- 性能瓶颈常来自频繁传输，优化思路包括减少传输次数和批量处理。

## 10. C API 与 C++ 头文件接口
- `cl.h`：过程式 C API，资源释放通常手动处理。
- `cl.hpp` / `opencl.hpp`：C++ 封装接口，常用 RAII 管理对象生命周期。
- 本仓库中：
  - `tasks/` 使用 C API 风格；
  - `tasks_cpp/` 使用 C++ 头文件接口进行同主题练习。
