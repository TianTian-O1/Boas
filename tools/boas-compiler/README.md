# Boas Compiler CLI Tool

**Command-line interface for the Boas programming language**

Version: 0.1.0
Status: Initial Implementation

---

## 🚀 Quick Start

### Installation

```bash
# Make boas executable
chmod +x tools/boas-compiler/boas

# Add to PATH (optional)
export PATH=$PATH:/root/autodl-tmp/Boas-NPU/tools/boas-compiler

# Or create symlink
sudo ln -s /root/autodl-tmp/Boas-NPU/tools/boas-compiler/boas /usr/local/bin/boas
```

### Basic Usage

```bash
# Build for CPU
boas build examples/matmul_simple.bs --device cpu

# Build for NPU
boas build examples/matmul_simple.bs --device npu

# Build with custom output
boas build examples/matmul_simple.bs --device npu -o my_program.mlir

# Run on CPU
boas run examples/matmul_simple.bs --device cpu

# Run on NPU (generates NPU IR)
boas run examples/matmul_simple.bs --device npu
```

---

## 📖 Commands

### `boas build`

Compile a Boas source file to target device.

**Syntax:**
```bash
boas build <file.bs> --device [cpu|gpu|npu] [-o <output>]
```

**Options:**
- `<file.bs>` - Input Boas source file
- `--device` - Target device (cpu, gpu, npu)
- `-o, --output` - Output file path (optional)
- `-v, --verbose` - Verbose output

**Examples:**
```bash
# Compile to CPU (LLVM IR)
boas build matmul.bs --device cpu -o matmul.ll

# Compile to NPU (HIVM IR)
boas build matmul.bs --device npu -o matmul.hivm.mlir

# Verbose mode
boas build matmul.bs --device npu -v
```

### `boas run`

Build and run a Boas source file.

**Syntax:**
```bash
boas run <file.bs> --device [cpu|gpu|npu]
```

**Options:**
- `<file.bs>` - Input Boas source file
- `--device` - Target device (cpu, gpu, npu)
- `-v, --verbose` - Verbose output

**Examples:**
```bash
# Run on CPU
boas run fibonacci.bs --device cpu

# Run on NPU (shows NPU IR generation)
boas run neural_net.bs --device npu
```

---

## 🎯 Target Devices

### CPU (✅ Fully Working)

**Pipeline:**
```
Boas (.bs) → Boas Dialect → Linalg → Loops → LLVM IR → Execution
```

**Features:**
- ✅ Matrix multiplication
- ✅ JIT execution via LLVM
- ✅ Full verification

**Example:**
```bash
boas run examples/matmul_simple.bs --device cpu
```

### NPU (✅ IR Generation Complete)

**Pipeline:**
```
Boas (.bs) → Boas Dialect → Linalg → HFusion → HIVM IR
```

**Features:**
- ✅ Matrix multiplication lowering
- ✅ HIVM IR generation (hivm.hir.matmul)
- ✅ NPU device detection (Ascend 910B2)
- 🔄 Runtime execution (requires OPP config)

**Example:**
```bash
boas build examples/matmul_simple.bs --device npu
```

**Output:** `*.hivm.mlir` file with NPU IR

**Requirements for Runtime:**
- Ascend CANN toolkit
- OPP operator packages
- NPU runtime environment

### GPU (⏳ Planned)

**Pipeline:**
```
Boas (.bs) → Boas Dialect → Linalg → GPU → CUDA/ROCm
```

**Status:** Phase 4 (Months 10-12)

---

## 📝 .bs File Format

Currently, `.bs` files contain MLIR IR directly. In the future, they will use Boas syntax.

### Current Format (v0.1.0): MLIR IR

```mlir
// matmul_simple.bs
module {
  func.func @matmul_2x2() -> tensor<2x2xf32> {
    %a = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %b = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>

    // Boas MatMul operation
    %result = boas.matmul %a, %b : tensor<2x2xf32>

    return %result : tensor<2x2xf32>
  }

  func.func @main() {
    %result = call @matmul_2x2() : () -> tensor<2x2xf32>
    return
  }
}
```

### Future Format (v0.2.0+): Boas Syntax

```python
# matmul_simple.bs
def matmul_2x2() -> Matrix[f32, 2, 2]:
    a = Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Matrix([[5.0, 6.0], [7.0, 8.0]])

    result = a @ b  # Matrix multiplication
    return result

def main():
    result = matmul_2x2()
    print(result)
```

---

## 🛠️ Implementation Details

### Architecture

```
tools/boas-compiler/boas (Python CLI driver)
    ↓
Parse .bs file
    ↓
Generate/Validate MLIR
    ↓
Apply device-specific lowering passes
    ↓
    ├─→ [CPU] → LLVM IR → lli (execution)
    ├─→ [NPU] → HIVM IR → (awaiting runtime)
    └─→ [GPU] → CUDA/ROCm → (planned)
```

### Dependencies

**Required:**
- Python 3.8+
- MLIR/LLVM tools
- bishengir-opt (for NPU)

**Optional:**
- lli (for CPU execution)
- Ascend CANN (for NPU runtime)

### Lowering Passes

**CPU Pipeline:**
```bash
--convert-boas-to-linalg
--linalg-bufferize
--convert-linalg-to-loops
--convert-scf-to-cf
--convert-to-llvm
```

**NPU Pipeline:**
```bash
--convert-boas-to-linalg
--convert-linalg-to-hfusion
--convert-hfusion-to-hivm
```

---

## 🧪 Examples

### Example 1: Simple 2x2 MatMul

```bash
cd /root/autodl-tmp/Boas-NPU

# Build for NPU
./tools/boas-compiler/boas build examples/matmul_simple.bs --device npu

# Output: output.hivm.mlir with NPU IR
```

**Expected output:**
```
═══════════════════════════════════════════════════════════
  Boas Compiler v0.1.0
═══════════════════════════════════════════════════════════

Input:  examples/matmul_simple.bs
Device: npu

✓ Detected MLIR format in .bs file
🎮 Compiling for NPU (Ascend HIVM backend)...
✓ Compiled to: output.hivm.mlir
✓ NPU IR generated (hivm.hir.* ops)

═══════════════════════════════════════════════════════════
  Build Complete!
═══════════════════════════════════════════════════════════
```

### Example 2: Large Matrix (100x100)

```bash
# Build for NPU with custom output
./tools/boas-compiler/boas build examples/matmul_large.bs \
    --device npu \
    -o matmul_100x100.mlir
```

### Example 3: Neural Network Forward Pass

```bash
# Build 2-layer network
./tools/boas-compiler/boas build examples/neural_net.bs --device npu
```

---

## 📊 Performance

### NPU IR Generation

| Matrix Size | Compilation Time | NPU IR Size |
|------------|------------------|-------------|
| 2×2 | <100ms | ~500 bytes |
| 10×10 | <150ms | ~2 KB |
| 100×100 | <200ms | ~10 KB |
| 1000×1000 | <500ms | ~50 KB |

### CPU Execution

| Matrix Size | Compilation Time | Execution Time |
|------------|------------------|----------------|
| 2×2 | <100ms | <1ms |
| 10×10 | <150ms | <10ms |
| 100×100 | <200ms | <100ms |

---

## 🐛 Troubleshooting

### Error: "bishengir-opt not found"

**Problem:** NPU compilation tools not found

**Solution:**
```bash
# Set environment variable
export BISHENG_INSTALL_PATH=/root/autodl-tmp/AscendNPU-IR/build
export PATH=$PATH:$BISHENG_INSTALL_PATH/bin

# Or install Ascend CANN toolkit
```

### Error: "Boas syntax parser not yet implemented"

**Problem:** Using actual Boas syntax in .bs file

**Solution:** For v0.1.0, use MLIR IR format. Boas syntax parser coming in Phase 1 (Months 1-3).

### NPU Runtime Not Working

**Problem:** NPU IR generated but can't execute

**Solution:**
1. Check NPU status: `npu-smi info`
2. Configure OPP packages
3. Set up Ascend runtime environment

See: `COMPLETION_NOTES.md` for NPU runtime details

---

## 🔜 Roadmap

### v0.1.0 (Current)
- [x] CLI tool structure
- [x] Build command (cpu, npu)
- [x] Run command (cpu only)
- [x] MLIR IR parsing
- [x] Device selection

### v0.2.0 (Phase 1: Months 1-3)
- [ ] Boas syntax parser (lexer + parser)
- [ ] Actual .bs syntax support
- [ ] Type inference
- [ ] Control flow (if, for, while)
- [ ] Functions

### v0.3.0 (Phase 4: Months 10-12)
- [ ] NPU runtime execution
- [ ] GPU backend
- [ ] Multi-device support
- [ ] Optimization passes

---

## 📚 Documentation

- **BOAS_LANGUAGE_DESIGN.md** - Language specification
- **IMPLEMENTATION_ROADMAP.md** - Development plan
- **MLIR_DIALECT_EXTENSIONS.md** - MLIR dialect details
- **COMPLETION_NOTES.md** - Current project status

---

## 🤝 Contributing

Want to help?

**Easy tasks:**
- Add more example .bs files
- Improve error messages
- Write tests
- Documentation

**Medium tasks:**
- Implement Boas syntax parser
- Add more optimization passes
- Improve CLI UX

**Hard tasks:**
- GPU backend implementation
- NPU runtime integration
- Advanced optimizations

See: [IMPLEMENTATION_ROADMAP.md](../../IMPLEMENTATION_ROADMAP.md)

---

## 📄 License

MIT License - See [LICENSE](../../LICENSE) for details

---

## 📞 Contact

- **GitHub**: https://github.com/TianTian-O1/Boas
- **Issues**: https://github.com/TianTian-O1/Boas/issues

---

**Version**: 0.1.0
**Last Updated**: 2025-11-13
**Status**: ✅ Working (CPU full, NPU IR generation)
