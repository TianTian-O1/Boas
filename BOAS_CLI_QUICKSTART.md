# Boas CLI Tool - Quick Start Guide

**First working version of the `boas` command-line compiler!**

Version: 0.1.0
Status: ✅ Working

---

## 🚀 Installation

```bash
# The boas tool is in the project root
cd /root/autodl-tmp/Boas-NPU

# Make it executable (already done)
chmod +x boas

# Test it
./boas --help
```

---

## 📖 Usage

### Build Command

Compile .bs files to target device:

```bash
# Build for NPU
./boas build examples/matmul_simple.bs --device npu

# Build for CPU
./boas build examples/matmul_simple.bs --device cpu

# Custom output file
./boas build examples/matmul_large.bs --device npu -o my_program.mlir
```

### Run Command

Build and "run" (currently shows IR):

```bash
./boas run examples/matmul_simple.bs --device npu
```

---

## 📝 Examples

### Example 1: Simple MatMul (2x2)

```bash
./boas build examples/matmul_simple.bs --device npu
```

**Output:**
- Generates Linalg IR showing:
  - tensor.empty() - Output allocation
  - linalg.fill - Zero initialization
  - linalg.matmul - Matrix multiplication

### Example 2: Large Matrix (100x100)

```bash
./boas build examples/matmul_large.bs --device npu -o large.mlir
```

### Example 3: Neural Network

```bash
./boas build examples/neural_net.bs --device cpu
```

---

## ✅ What Works

**v0.1.0:**
- ✅ `boas build` command
- ✅ `boas run` command
- ✅ Device selection (--device cpu/npu)
- ✅ Custom output (-o)
- ✅ Boas → Linalg conversion (working!)
- ✅ 3 example .bs files

**Currently Shows:**
- Linalg IR (intermediate representation)
- Proper tensor operations
- Correct matrix dimensions

---

## 🔄 Next Steps

**Phase 1 (Next):**
- [ ] Integrate full MLIR pipeline
- [ ] Add Linalg → LLVM for CPU
- [ ] Add Linalg → HIVM for NPU
- [ ] Actual execution support

**Current Status:**
- Demonstrates compilation concept ✅
- Shows IR generation ✅
- Validates tool structure ✅

---

## 📊 Generated IR Example

```mlir
module {
  func.func @matmul_2x3_3x4(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = tensor.empty() : tensor<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%1 : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
}
```

---

## 🎯 Testing

```bash
# Test all examples
for file in examples/*.bs; do
    echo "Testing: $file"
    ./boas build "$file" --device npu
    echo ""
done
```

---

## 📚 Documentation

- `tools/boas-compiler/README.md` - Full CLI documentation
- `BOAS_LANGUAGE_DESIGN.md` - Language specification
- `IMPLEMENTATION_ROADMAP.md` - Development plan

---

**Status**: ✅ Working prototype
**Next**: Integrate full MLIR passes
**GitHub**: https://github.com/TianTian-O1/Boas
