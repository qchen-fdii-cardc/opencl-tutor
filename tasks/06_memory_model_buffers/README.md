# 任务 06: 内存模型与缓冲区

## 任务目标
使用不同内存标志创建 Buffer，完成 Host/Device 数据传输。

## 核心概念
CL_MEM_READ_ONLY/WRITE_ONLY/READ_WRITE、映射与拷贝。

## 代码实现说明
当前目录提供构建骨架与占位代码，后续将在本任务中逐步完善。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\\build\\Debug\\task_06.exe
```

## 验收标准
演示不同内存标志下的读写行为。

