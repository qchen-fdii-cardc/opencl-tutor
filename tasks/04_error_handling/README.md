# 任务 04: 统一错误处理机制

## 任务目标
实现一套可复用的 OpenCL 错误检查机制（`CL_CHECK` + 错误码转字符串）。

## 学习动机
OpenCL API 大多返回 `cl_int`，如果每处手写判断，代码会快速失控。统一错误处理是工程化关键。

## 关键概念
- **错误码映射**：数字错误码转可读字符串。
- **失败即中断**：关键路径失败直接抛异常。
- **最小重复原则**：统一入口减少重复样板。

## 关键函数与常量
- `check_cl(err, what)`：统一检查逻辑。
- `cl_error_to_string`：错误码到文本映射。
- `CL_CHECK(call)`：宏包装 API 调用。
- 常见错误常量：`CL_INVALID_VALUE`、`CL_INVALID_DEVICE` 等。

## 代码实现思路
1. 定义错误码文本映射函数。
2. 封装检查函数，失败抛 `std::runtime_error`。
3. 用 `CL_CHECK` 包装平台/设备查询与资源创建。
4. 在 `main` 中用 `try/catch` 汇总处理。

## 构建与运行
```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\task_04.exe
```

## 验收标准
- 成功路径输出 `Unified error handling flow completed successfully.`。
- 失败路径输出明确 API 名称和错误码文本。

## 常见错误与排查
- 输出 `CL_UNKNOWN_ERROR`：补充映射表。
- 程序异常退出：确认 `catch` 覆盖 `std::exception`。
