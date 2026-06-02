# 任务 02: 创建上下文与命令队列

## 任务目标
为首个可用设备创建 OpenCL 上下文和命令队列。

## 核心概念
上下文（Context）、命令队列（Command Queue）、资源生命周期。

## 代码实现说明
当前目录提供构建骨架与占位代码，后续将在本任务中逐步完善。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\\build\\Debug\\task_02.exe
```

## 验收标准
可成功创建并释放 Context/Queue。

