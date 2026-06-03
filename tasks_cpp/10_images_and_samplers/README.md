# 任务 10: OpenCL 图像与采样器

## 任务目标
使用 `image2d_t + sampler` 完成 RGBA 图像拷贝。

## 学习动机
图像对象是 OpenCL 区别于普通 GPGPU Buffer 编程的重要能力，常用于图像/视频管线。

## 关键概念
- **Image Object**：带格式与维度语义的内存对象。
- **Sampler**：定义坐标归一化、边界模式、过滤方式。
- **read_image / write_image**：图像读写内建函数。

## 关键函数与常量
- `cl::Image2D`
- `queue.enqueueReadImage(...)`
- `CL_MEM_OBJECT_IMAGE2D`
- `CL_RGBA` / `CL_FLOAT`
- `CLK_NORMALIZED_COORDS_FALSE` / `CLK_ADDRESS_CLAMP` / `CLK_FILTER_NEAREST`

## 代码实现思路
1. 构造测试图像（RGBA）。
2. 创建输入/输出 `image2d_t`。
3. 执行 `copy_image` 内核：读取并原样写回。
4. 读回图像并逐像素校验。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_10.exe
```

## 验收标准
- 输出 `Image sampler task verified.`。
- 所有像素满足 `dst == src`（浮点误差阈值内）。

## 常见错误与排查
- 创建图像失败：检查 `cl_image_format` 与设备支持。
- 颜色错误：检查 `read_imageui/write_imageui` 与数据类型匹配。
