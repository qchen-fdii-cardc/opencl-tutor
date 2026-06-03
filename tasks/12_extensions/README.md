# 任务 12: 设备扩展能力探索

## 任务目标
枚举设备支持的 OpenCL 扩展和 OpenCL C 版本，建立“设备能力画像”。

## 学习动机
不同厂商设备能力差异明显，扩展查询是写可移植代码前的必要步骤。

## 关键概念
- **设备能力探测**：运行时而非硬编码假设。
- **扩展字符串**：空格分隔的扩展名列表。
- **特性门控**：根据扩展决定是否启用高级路径。

## 关键函数与常量
- `clGetDeviceInfo(..., CL_DEVICE_EXTENSIONS, ...)`
- `clGetDeviceInfo(..., CL_DEVICE_OPENCL_C_VERSION, ...)`
- `CL_DEVICE_NAME`

## 代码实现思路
1. 枚举平台与设备。
2. 查询设备名、OpenCL C 版本。
3. 读取扩展字符串并按空格拆分打印。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_12.exe
```

## 验收标准
- 每个设备至少打印设备名与 OpenCL C 版本。
- 扩展列表逐行展示，便于人工筛选。

## 常见错误与排查
- 扩展为空：确认设备可见且查询参数正确。
- 字符串乱码：采用“两步查询长度再读取”模式。
