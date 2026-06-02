# 任务 11: 高斯模糊图像处理

## 任务目标
实现 OpenCL 高斯模糊并与 CPU 版本比较。

## 核心概念
卷积、边界处理、图像访存优化。

## 代码实现说明
当前目录提供构建骨架与占位代码，后续将在本任务中逐步完善。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\\build\\Debug\\task_11.exe
```

## 验收标准
输出图像正确且有性能对比。

