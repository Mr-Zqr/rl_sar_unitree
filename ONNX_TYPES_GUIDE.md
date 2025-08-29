## ONNX 模型输出变量类型说明

ONNX 模型的输出变量类型由 `ONNXTensorElementDataType` 枚举定义，主要包括以下类型：

### 常见数据类型：

1. **FLOAT32** (`ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`)
   - 32位浮点数，最常用的类型
   - 大小：4 字节/元素
   - 访问方法：`tensor.GetTensorMutableData<float>()`

2. **DOUBLE** (`ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE`)  
   - 64位双精度浮点数
   - 大小：8 字节/元素
   - 访问方法：`tensor.GetTensorMutableData<double>()`

3. **INT32** (`ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32`)
   - 32位有符号整数
   - 大小：4 字节/元素  
   - 访问方法：`tensor.GetTensorMutableData<int32_t>()`

4. **INT64** (`ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64`)
   - 64位有符号整数
   - 大小：8 字节/元素
   - 访问方法：`tensor.GetTensorMutableData<int64_t>()`

### 其他支持的类型：

- **UINT8** - 8位无符号整数
- **INT8** - 8位有符号整数  
- **UINT16** - 16位无符号整数
- **INT16** - 16位有符号整数
- **UINT32** - 32位无符号整数
- **UINT64** - 64位无符号整数
- **FLOAT16** - 16位半精度浮点数
- **BFLOAT16** - Brain Float 16位浮点数
- **BOOL** - 布尔类型
- **STRING** - 字符串类型
- **COMPLEX64** - 64位复数
- **COMPLEX128** - 128位复数

### 使用方法：

```cpp
// 获取输出张量
auto outputs = engine.FirstOutput();

// 检查每个输出的类型
for (size_t i = 0; i < outputs.size(); ++i) {
    // 获取类型枚举
    ONNXTensorElementDataType type = ONNXInferenceEngine::GetTensorDataType(outputs[i]);
    
    // 获取类型字符串
    std::string type_str = ONNXInferenceEngine::GetTensorDataTypeString(outputs[i]);
    
    // 根据类型访问数据
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            {
                float* data = outputs[i].GetTensorMutableData<float>();
                // 处理 float 数据...
            }
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            {
                double* data = outputs[i].GetTensorMutableData<double>();
                // 处理 double 数据...
            }
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            {
                int64_t* data = outputs[i].GetTensorMutableData<int64_t>();
                // 处理 int64 数据...
            }
            break;
        // 其他类型...
    }
}
```

### 注意事项：

1. **类型匹配**：调用 `GetTensorMutableData<T>()` 时，模板参数 `T` 必须与实际的张量数据类型匹配
2. **内存布局**：所有数据都是连续存储的，可以直接用指针访问
3. **生命周期**：张量数据的生命周期与 `Ort::Value` 对象绑定
4. **大多数 ML 模型**：通常输出 FLOAT32 类型，但也可能有 INT64（如索引）或其他类型

### 在代码中的应用：

当前的 `ONNXInferenceEngine` 类提供了以下辅助方法来处理不同类型：

- `GetTensorDataType()` - 获取类型枚举
- `GetTensorDataTypeString()` - 获取类型字符串描述  
- `ExtractTensorData()` - 提取为 float vector（仅适用于 float 类型）
- `GetTensorShape()` - 获取张量形状
- `GetTensorElementCount()` - 获取元素总数
