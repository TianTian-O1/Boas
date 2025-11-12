#!/bin/bash
# Quick demo of the Boas CLI tool

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         Boas CLI Tool Demo                                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Set up environment
export PATH=$PATH:/root/autodl-tmp/Boas-NPU/tools/boas-compiler
cd /root/autodl-tmp/Boas-NPU

echo "1. Show help"
echo "─────────────────────────────────────────────────────────────"
./tools/boas-compiler/boas --help
echo ""
echo ""

echo "2. List example .bs files"
echo "─────────────────────────────────────────────────────────────"
ls -lh examples/*.bs
echo ""
echo ""

echo "3. Show example file content"
echo "─────────────────────────────────────────────────────────────"
echo "examples/matmul_simple.bs:"
head -20 examples/matmul_simple.bs
echo ""
echo ""

echo "4. Build for NPU (RECOMMENDED)"
echo "─────────────────────────────────────────────────────────────"
echo "Command: boas build examples/matmul_simple.bs --device npu"
echo ""

# Check if bishengir-opt is available
if [ -f "/root/autodl-tmp/AscendNPU-IR/build/bin/bishengir-opt" ]; then
    echo "✓ bishengir-opt found - Running NPU build..."
    ./tools/boas-compiler/boas build examples/matmul_simple.bs --device npu -o demo_npu.mlir
    echo ""
    if [ -f "demo_npu.mlir" ]; then
        echo "✓ Generated NPU IR:"
        echo "  File: demo_npu.mlir"
        echo "  Size: $(wc -c < demo_npu.mlir) bytes"
        echo ""
        echo "  First 30 lines:"
        head -30 demo_npu.mlir
    fi
else
    echo "⚠️  bishengir-opt not found"
    echo "   NPU compilation requires: /root/autodl-tmp/AscendNPU-IR/build/bin/bishengir-opt"
    echo "   Skipping NPU build demo"
fi

echo ""
echo ""

echo "5. Test with other examples"
echo "─────────────────────────────────────────────────────────────"
echo ""
echo "Try these commands:"
echo ""
echo "  # Large matrix (100x100)"
echo "  boas build examples/matmul_large.bs --device npu"
echo ""
echo "  # Neural network forward pass"
echo "  boas build examples/neural_net.bs --device npu"
echo ""
echo "  # Verbose mode"
echo "  boas build examples/matmul_simple.bs --device npu -v"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  Demo Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📚 Documentation: tools/boas-compiler/README.md"
echo "🔗 GitHub: https://github.com/TianTian-O1/Boas"
echo ""
