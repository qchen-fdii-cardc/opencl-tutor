# 任务 04: 统一错误处理

## 任务目标
为 OpenCL API 调用建立统一错误处理与可读错误信息。

## 核心概念
错误码映射、宏封装、失败即退出策略。

## 代码实现说明
当前目录提供构建骨架与占位代码，后续将在本任务中逐步完善。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\\build\\Debug\\task_04.exe
```

## 验收标准
任意 API 失败时可打印可读错误上下文。

