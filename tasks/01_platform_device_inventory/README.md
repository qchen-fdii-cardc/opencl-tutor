# 任务 01: 列出平台与设备

## 任务目标
枚举系统中的 OpenCL 平台和设备，打印平台名、设备名与设备类型。

## 核心概念
平台（Platform）、设备（Device）、查询 API（clGetPlatformIDs/clGetDeviceIDs）。

## 代码实现说明
本任务已实现完整示例代码。请阅读 src/main.cpp 了解平台与设备枚举流程。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\\build\\Debug\\task_01.exe
```

## 验收标准
程序正常打印至少一个平台信息；如果无平台，给出清晰提示。

