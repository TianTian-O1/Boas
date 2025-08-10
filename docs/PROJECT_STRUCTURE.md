# 📁 Boas项目文件结构

## 🎯 **重新组织后的清晰结构**

```
Boas-linux/
├── 📁 benchmark/           # 性能测试套件
│   ├── src/               # 测试源码
│   ├── scripts/          # 测试脚本  
│   ├── results/          # 测试结果
│   └── *.py              # Python测试工具
├── 📁 build/              # CMake构建输出
├── 📁 docs/               # 📚 文档目录
│   ├── design/           # 设计文档
│   ├── performance/      # 性能分析
│   └── reports/          # 测试报告
├── 📁 examples/           # 🎯 示例代码
├── 📁 include/            # C++头文件
│   ├── frontend/         # 前端解析器
│   └── mlirops/          # MLIR操作
├── 📁 lib/               # C++源文件  
│   ├── frontend/         # 前端实现
│   ├── mlirops/          # MLIR实现
│   └── runtime/          # 运行时库
├── 📁 results/            # 📊 测试结果数据
├── 📁 scripts/            # 🔧 工具脚本
├── 📁 test/              # 基础测试
├── 📁 tests/             # 🧪 组织化测试
│   ├── unit/            # 单元测试
│   ├── integration/     # 集成测试
│   └── npu/            # NPU专项测试
├── 📁 tools/             # 🛠️ 开发工具
│   └── visualization/   # 可视化工具
├── CMakeLists.txt        # 构建配置
└── README.md            # 项目说明
```

## 📂 **目录职责说明**

### 🏗️ **核心源码**
- `include/` + `lib/`: Boas编译器核心实现
- `CMakeLists.txt`: 构建系统配置

### 📚 **文档系统**  
- `docs/design/`: 架构设计文档
- `docs/performance/`: 性能分析文档
- `docs/reports/`: 测试和实现报告

### 🧪 **测试体系**
- `test/`: 基础功能测试
- `tests/unit/`: 单元测试
- `tests/integration/`: 集成测试
- `tests/npu/`: NPU专项测试
- `benchmark/`: 性能基准测试

### 🎯 **示例和工具**
- `examples/`: 使用示例
- `scripts/`: 构建和部署脚本
- `tools/visualization/`: 性能可视化工具

### 📊 **结果数据**
- `results/`: JSON格式的测试结果
- `benchmark/results/`: 基准测试结果

## 🚀 **使用指南**

### 构建项目
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行测试
```bash
# NPU功能测试
./tests/npu/test_cann_simple.bs

# 性能基准测试  
cd benchmark && python3 simple_benchmark.py

# 生成性能图表
cd tools/visualization && python3 generate_performance_charts.py
```

### 查看文档
```bash
# 查看性能分析报告
cat docs/reports/PERFORMANCE_ANALYSIS_REPORT.md

# 查看技术实现状态
cat docs/reports/CANN_INTEGRATION_SUCCESS.md
```

## 🎯 **文件命名规范**

### 📄 **文档文件**
- `*.md`: Markdown文档
- 前缀规范:
  - `PERFORMANCE_*`: 性能相关
  - `NPU_*`: NPU适配相关  
  - `CANN_*`: CANN集成相关

### 🧪 **测试文件**
- `test_*.bs`: Boas语言测试
- `test_*.py`: Python测试脚本
- `*_benchmark.py`: 性能测试

### 🔧 **工具脚本**
- `*.sh`: Shell脚本
- `generate_*.py`: 生成工具
- `monitor_*.py`: 监控工具

## ✅ **清理完成的效果**

### 🗑️ **删除的冗余文件**
- 重复的测试脚本
- 过时的临时文件  
- 散乱的文档文件

### 📁 **重新组织的文件**
- 33个根目录散乱文件 → 清晰的目录结构
- 文档集中到 `docs/`
- 测试集中到 `tests/`  
- 工具集中到 `tools/`

### 🎯 **便于开发的结构**
- 功能模块清晰分离
- 文档容易查找
- 测试分类明确
- 工具统一管理

---

*此结构遵循现代C++项目的最佳实践，便于团队协作和项目维护。*
