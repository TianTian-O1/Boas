#!/bin/bash

echo "=== Boas NPU环境配置脚本 ==="

# 检查并设置CANN环境
setup_cann_env() {
    echo "检查CANN环境..."
    
    # 检查CANN安装
    CANN_PATHS=(
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        "$HOME/Ascend/ascend-toolkit/set_env.sh" 
        "/opt/ascend/ascend-toolkit/set_env.sh"
    )
    
    CANN_FOUND=false
    for path in "${CANN_PATHS[@]}"; do
        if [ -f "$path" ]; then
            echo "找到CANN环境文件: $path"
            source "$path"
            CANN_FOUND=true
            break
        fi
    done
    
    if [ "$CANN_FOUND" = false ]; then
        echo "警告: 未找到CANN环境，请确保已正确安装CANN toolkit"
        echo "请从昇腾社区下载并安装CANN 8.2.RC1.alpha003版本"
        echo "下载地址: https://www.hiascend.com/developer/download/community/result?module=cann"
        return 1
    fi
    
    echo "CANN环境配置成功"
    return 0
}

# 检查NPU设备
check_npu_devices() {
    echo "检查NPU设备..."
    
    # 检查设备文件
    if [ -c "/dev/davinci0" ]; then
        echo "找到NPU设备: /dev/davinci0"
        ls -la /dev/davinci*
    elif [ -c "/dev/davinci_manager" ]; then
        echo "找到NPU管理器: /dev/davinci_manager"
    else
        echo "警告: 未找到NPU设备文件"
        echo "请检查NPU驱动是否正确安装"
    fi
    
    # 检查npu-smi工具
    if command -v npu-smi >/dev/null 2>&1; then
        echo "NPU设备信息:"
        npu-smi info
    else
        echo "提示: npu-smi工具未找到，无法显示详细设备信息"
    fi
}

# 检查Python环境
check_python_env() {
    echo "检查Python环境..."
    
    # 检查Python版本
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "Python版本: $python_version"
    
    # 检查torch和torch_npu
    if python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>/dev/null; then
        echo "PyTorch已安装"
    else
        echo "警告: PyTorch未安装"
        echo "请安装: pip install torch==2.6.0"
    fi
    
    if python3 -c "import torch_npu; print(f'torch_npu版本: {torch_npu.__version__}')" 2>/dev/null; then
        echo "torch_npu已安装"
    else
        echo "警告: torch_npu未安装"
        echo "请安装: pip install torch_npu==2.6.0"
    fi
    
    # 检查triton（可选）
    if python3 -c "import triton; print(f'Triton版本: {triton.__version__}')" 2>/dev/null; then
        echo "Triton已安装（用于高性能kernel开发）"
    else
        echo "提示: Triton未安装（可选，用于自定义高性能kernel）"
    fi
}

# 生成NPU测试脚本
generate_test_script() {
    echo "生成NPU测试脚本..."
    
    cat > test_npu_basic.py << 'EOF'
#!/usr/bin/env python3
"""
基础NPU功能测试
"""

import sys

def test_torch_npu():
    try:
        import torch
        import torch_npu
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"torch_npu版本: {torch_npu.__version__}")
        
        if torch.npu.is_available():
            device_count = torch.npu.device_count()
            print(f"NPU设备数量: {device_count}")
            
            for i in range(device_count):
                torch.npu.set_device(i)
                print(f"NPU设备 {i}: {torch.npu.get_device_name(i)}")
            
            # 简单的矩阵运算测试
            print("执行基础矩阵运算测试...")
            torch.npu.set_device(0)
            a = torch.randn(100, 100, device='npu')
            b = torch.randn(100, 100, device='npu')
            c = torch.matmul(a, b)
            print("NPU矩阵乘法测试通过")
            
            return True
        else:
            print("错误: NPU设备不可用")
            return False
            
    except ImportError as e:
        print(f"导入错误: {e}")
        return False
    except Exception as e:
        print(f"NPU测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== NPU基础功能测试 ===")
    if test_torch_npu():
        print("NPU环境测试通过!")
        sys.exit(0)
    else:
        print("NPU环境测试失败!")
        sys.exit(1)
EOF

    chmod +x test_npu_basic.py
    echo "NPU测试脚本已生成: test_npu_basic.py"
}

# 主函数
main() {
    echo "开始配置Boas NPU环境..."
    
    # 设置CANN环境
    if ! setup_cann_env; then
        echo "CANN环境配置失败，但继续其他检查..."
    fi
    
    # 检查NPU设备
    check_npu_devices
    
    # 检查Python环境
    check_python_env
    
    # 生成测试脚本
    generate_test_script
    
    echo ""
    echo "=== 环境配置完成 ==="
    echo "请执行以下命令测试NPU环境:"
    echo "  python3 test_npu_basic.py"
    echo ""
    echo "如果测试通过，可以使用以下命令测试Boas NPU矩阵乘法:"
    echo "  ./run.sh -d all -t test/test_npu_matmul.bs"
}

# 执行主函数
main "$@"
