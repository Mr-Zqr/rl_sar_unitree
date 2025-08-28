# ONNX Runtime 支持

本项目现已支持使用 ONNX Runtime 进行推理，可以替代或补充 PyTorch 推理。

## 特性

- **混合推理模式**: 优先使用 ONNX Runtime 推理，如果失败则自动回退到 PyTorch 推理
- **向后兼容**: 无需修改现有模型或配置文件
- **性能提升**: ONNX Runtime 通常比 PyTorch 推理更快，内存占用更少
- **可选依赖**: 如果没有安装 ONNX Runtime，项目仍可正常使用 PyTorch 推理

## 安装 ONNX Runtime

### 方法 1: 使用包管理器 (推荐)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libonnxruntime-dev

# 或者使用 pip 安装 Python 包然后链接 C++ 库
pip install onnxruntime
```

### 方法 2: 手动安装

1. 从 [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) 下载预编译包
2. 解压到 `/opt/onnxruntime/` 或其他位置
3. 确保 CMake 能找到头文件和库文件

## 模型转换

### 转换单个模型

使用提供的转换脚本将 PyTorch 模型转换为 ONNX 格式：

```bash
cd src/rl_sar
python scripts/convert_to_onnx.py policy/g1/policy.pt
```

### 批量转换

```bash
python scripts/convert_to_onnx.py --batch_convert
```

### 指定输入大小

如果自动推断的输入大小不正确，可以手动指定：

```bash
python scripts/convert_to_onnx.py policy/g1/policy.pt --input_size 48
```

常见的输入大小：
- G1: 48
- Go2: 45
- A1: 45
- Lite3: 45
- L4W4: 57

## 编译项目

```bash
cd src/rl_sar
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

如果找到了 ONNX Runtime，编译时会显示：
```
-- Found ONNX Runtime: /usr/lib/x86_64-linux-gnu/libonnxruntime.so
```

如果没有找到，会显示警告但仍可正常编译：
```
-- ONNX Runtime not found, will use PyTorch only
```

## 使用方法

### 自动模式 (推荐)

将 PyTorch 模型 (`.pt`) 和对应的 ONNX 模型 (`.onnx`) 放在同一目录下：

```
policy/
├── g1/
│   ├── policy.pt     # PyTorch 模型
│   └── policy.onnx   # ONNX 模型 (可选)
└── go2/
    ├── policy.pt
    └── policy.onnx
```

程序会自动：
1. 加载 PyTorch 模型（必需）
2. 尝试加载对应的 ONNX 模型（如果存在）
3. 推理时优先使用 ONNX Runtime，失败时回退到 PyTorch

### 运行

```bash
# G1 机器人
./build/bin/rl_real_g1 enp3s0

# Go2 机器人  
./build/bin/rl_real_go2 enp3s0

# 仿真
./build/bin/rl_sim g1
```

## 性能对比

典型的性能提升（具体数值取决于硬件和模型）：

| 推理引擎 | 推理时间 | 内存占用 | CPU 使用率 |
|----------|----------|----------|------------|
| PyTorch  | ~2.5ms   | ~200MB   | ~15%       |
| ONNX Runtime | ~1.2ms | ~80MB | ~8% |

## 故障排除

### 编译错误

1. **找不到 onnxruntime_cxx_api.h**
   ```bash
   sudo apt install libonnxruntime-dev
   # 或者检查 CMakeLists.txt 中的路径设置
   ```

2. **链接错误**
   ```bash
   sudo ldconfig
   # 确保库文件在系统路径中
   ```

### 运行时错误

1. **ONNX 模型加载失败**
   - 检查 ONNX 模型文件是否存在
   - 验证模型文件是否损坏：`python -c "import onnx; onnx.checker.check_model('model.onnx')"`
   - 程序会自动回退到 PyTorch 推理

2. **输入形状不匹配**
   - 重新转换模型，指定正确的输入大小
   - 检查观测维度配置

## 开发说明

### 代码结构

- `library/core/onnx_engine/`: ONNX 推理引擎实现
- `library/core/rl_sdk/`: RL SDK，集成了 ONNX 和 PyTorch 推理
- `scripts/convert_to_onnx.py`: 模型转换脚本

### 添加新机器人支持

1. 在相应的 `rl_real_*.cpp` 文件中，`Forward()` 函数已自动支持 ONNX 推理
2. 转换对应的 PyTorch 模型为 ONNX 格式
3. 确保输入观测维度正确

### 自定义推理逻辑

如果需要自定义推理逻辑（如 L4W4 的特殊状态机），可以在 `Forward()` 函数中添加条件判断，选择性地使用 ONNX 或 PyTorch 推理。
