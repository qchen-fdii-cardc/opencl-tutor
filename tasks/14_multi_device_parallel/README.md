# 任务 14: 多设备并行

## 任务目标
同时使用 CPU/GPU 等多设备并行计算并比较结果。

## 核心概念
多队列调度、任务切分、结果归并。

## 代码实现说明
当前目录提供构建骨架与占位代码，后续将在本任务中逐步完善。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\\build\\Debug\\task_14.exe
```

## 验收标准
多设备结果正确，给出性能与开销分析。

