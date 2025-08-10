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
