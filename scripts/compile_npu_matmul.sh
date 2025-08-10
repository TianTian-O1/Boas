#!/bin/bash

# BOAS NPU Matrix Multiplication Compilation and Execution Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "   BOAS NPU Matrix Multiplication Compiler"
echo "=============================================="

# Check if BiShengIR is available
BISHENGIR_PATH="/root/llvm-project/bishengir_aarch64"
if [ ! -d "$BISHENGIR_PATH" ]; then
    echo "Error: BiShengIR not found at $BISHENGIR_PATH"
    exit 1
fi

# Set environment variables
export PATH="$BISHENGIR_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$BISHENGIR_PATH/lib:$LD_LIBRARY_PATH"

# Check for Ascend environment
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "Warning: ASCEND_HOME_PATH not set, trying to detect..."
    if [ -d "/usr/local/Ascend/latest" ]; then
        export ASCEND_HOME_PATH="/usr/local/Ascend/latest"
        echo "Found Ascend at $ASCEND_HOME_PATH"
    elif [ -d "/root/Ascend/latest" ]; then
        export ASCEND_HOME_PATH="/root/Ascend/latest"
        echo "Found Ascend at $ASCEND_HOME_PATH"
    else
        echo "Warning: Ascend not found, NPU features may not work"
    fi
fi

# Function to compile BOAS file
compile_boas() {
    local input_file="$1"
    local output_file="${2:-${input_file%.bs}_npu}"
    
    echo ""
    echo "Compiling: $input_file"
    echo "Output: $output_file"
    
    # Step 1: Parse BOAS to MLIR
    echo "Step 1: Parsing BOAS to MLIR..."
    $PROJECT_ROOT/build/boas "$input_file" -emit=mlir -o temp.mlir
    
    # Step 2: Apply NPU optimizations
    echo "Step 2: Applying NPU optimizations..."
    $BISHENGIR_PATH/bin/bishengir-compile \
        temp.mlir \
        --convert-linalg-to-loops \
        --convert-scf-to-cf \
        --lower-affine \
        --convert-arith-to-llvm \
        --convert-memref-to-llvm \
        --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        --npu-matmul-opt \
        -o temp_optimized.mlir
    
    # Step 3: Generate LLVM IR
    echo "Step 3: Generating LLVM IR..."
    $BISHENGIR_PATH/bin/bishengir-compile \
        temp_optimized.mlir \
        --mlir-to-llvmir \
        -o temp.ll
    
    # Step 4: Compile to executable
    echo "Step 4: Compiling to executable..."
    clang++ -O3 -march=native \
        temp.ll \
        -L$PROJECT_ROOT/build/lib -lBoasRuntime \
        -L$BISHENGIR_PATH/lib -lBiShengIR \
        -o "$output_file"
    
    # Clean up temporary files
    rm -f temp.mlir temp_optimized.mlir temp.ll
    
    echo "✓ Compilation successful: $output_file"
}

# Function to run benchmarks
run_benchmark() {
    local executable="$1"
    
    echo ""
    echo "Running benchmark: $executable"
    echo "----------------------------------------"
    
    # Set NPU environment
    export NPU_DEVICE=0
    export NPU_VISIBLE_DEVICES=0
    
    # Run with performance monitoring
    if command -v npu-smi &> /dev/null; then
        echo "NPU Status before execution:"
        npu-smi info -t board -i 0 2>/dev/null || true
    fi
    
    # Execute the program
    time ./"$executable"
    
    if command -v npu-smi &> /dev/null; then
        echo ""
        echo "NPU Status after execution:"
        npu-smi info -t board -i 0 2>/dev/null || true
    fi
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    # Parse arguments
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <boas_file> [output_name]"
        echo ""
        echo "Examples:"
        echo "  $0 test/test_npu_matmul.bs"
        echo "  $0 test/test_npu_matmul_advanced.bs matmul_advanced"
        exit 1
    fi
    
    INPUT_FILE="$1"
    OUTPUT_NAME="${2:-}"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file not found: $INPUT_FILE"
        exit 1
    fi
    
    # Compile the BOAS file
    compile_boas "$INPUT_FILE" "$OUTPUT_NAME"
    
    # Ask to run the program
    echo ""
    read -p "Do you want to run the compiled program? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -n "$OUTPUT_NAME" ]; then
            run_benchmark "$OUTPUT_NAME"
        else
            run_benchmark "${INPUT_FILE%.bs}_npu"
        fi
    fi
    
    echo ""
    echo "=============================================="
    echo "              Process Complete"
    echo "=============================================="
}

# Run main function
main "$@"