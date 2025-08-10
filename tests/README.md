# Boas 测试套件

## 目录结构

```
tests/
├── unit/               # 单元测试
│   ├── cpp/           # C++ 单元测试
│   └── python/        # Python 单元测试
├── integration/        # 集成测试
├── npu/               # NPU 相关测试
├── boas/              # Boas 语言测试
│   └── matrix_tests/  # 矩阵运算测试
├── examples/          # 示例文件和MLIR测试
└── benchmarks/        # 性能基准测试
```

## 测试分类

### 单元测试 (unit/)
- **cpp/**: C++ 编写的底层功能测试
  - `test_ast.cpp` - AST 解析测试
  - `test_python_parser.cpp` - Python 解析器测试
  - `test_full_pipeline.cpp` - 完整编译流程测试
  
- **python/**: Python 编写的功能测试
  - `test_2x2_matrix.py` - 2x2 矩阵测试
  - `verify_boas_correctness.py` - Boas 正确性验证

### NPU 测试 (npu/)
包含所有 NPU（神经网络处理单元）相关的测试：
- `test_npu_*.bs` - NPU 功能测试
- `test_mixed_precision.bs` - 混合精度测试
- `monitor_npu_usage.py` - NPU 使用监控

### Boas 语言测试 (boas/)
Boas 语言特性和矩阵运算测试：
- `test_boas_dialect.bs` - Boas 方言测试
- `matrix_tests/` - 各种大小的矩阵运算测试

### 示例 (examples/)
- `boas_dialect_example.mlir` - MLIR 示例

## 运行测试

### 编译 C++ 测试
```bash
cd build
cmake ..
make -j$(nproc)
```

### 运行测试
```bash
# C++ 测试
./test-python-parser [测试文件]
./test-full-pipeline

# Python 测试
python tests/unit/python/test_2x2_matrix.py

# Boas 测试
./matrix-compiler --run tests/boas/test_boas_dialect.bs
```

## 添加新测试

1. 根据测试类型选择合适的目录
2. 遵循现有的命名规范：
   - C++ 测试: `test_*.cpp`
   - Python 测试: `test_*.py` 或 `*_test.py`
   - Boas 测试: `test_*.bs`
3. 如果是 C++ 测试，需要更新 CMakeLists.txt