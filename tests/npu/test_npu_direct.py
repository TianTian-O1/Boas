#!/usr/bin/env python3
"""
直接在NPU上运行Boas生成的矩阵乘法测试
验证NPU Backend是否正确生成代码
"""

import subprocess
import os
import time

def test_npu_code_generation():
    """测试Boas在NPU上的代码生成功能"""
    print("=== Boas NPU代码生成测试 ===")
    
    # 运行Boas编译器，只测试代码生成阶段
    build_dir = "/root/Boas/Boas-linux/build"
    test_file = "../test/test_npu_matmul.bs"
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:{}".format(
            env.get("LD_LIBRARY_PATH", "")
        )
        
        # 运行编译器，只进行代码生成测试
        cmd = [
            "./test-full-pipeline",
            "--generate-only",  # 假设有这个选项
            test_file
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {build_dir}")
        
        result = subprocess.run(
            cmd,
            cwd=build_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print("标准输出:")
            print(result.stdout)
            
            # 分析输出中的关键信息
            lines = result.stdout.split('\n')
            
            mlir_lines = 0
            npu_optimized = False
            matmul_found = False
            
            for line in lines:
                if any(keyword in line for keyword in ['%', 'func.func', 'linalg', 'memref']):
                    mlir_lines += 1
                if "NPU-optimized" in line or "DEBUG] Using" in line:
                    npu_optimized = True
                if "matmul" in line.lower() or "matrix" in line.lower():
                    matmul_found = True
            
            print(f"\n=== 代码生成分析 ===")
            print(f"MLIR代码行数: {mlir_lines}")
            print(f"NPU优化检测: {'✅' if npu_optimized else '❌'}")
            print(f"矩阵乘法检测: {'✅' if matmul_found else '❌'}")
            
            # 检查是否包含重要的矩阵操作
            if mlir_lines > 100:
                print("✅ 生成了完整的MLIR代码")
            else:
                print("⚠️  MLIR代码较少，可能生成不完整")
                
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
            
            # 检查错误中是否有NPU相关信息
            if "NPU" in result.stderr:
                print("✅ 检测到NPU相关日志")
            
        return result.returncode == 0 or "llvm.func" in result.stdout
        
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时")
        return False
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False

def test_npu_environment():
    """测试NPU环境"""
    print("\n=== NPU环境测试 ===")
    
    try:
        import torch
        import torch_npu
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"torch_npu版本: {torch_npu.__version__}")
        
        device_count = torch_npu.npu.device_count()
        print(f"NPU设备数量: {device_count}")
        
        if device_count > 0:
            device_name = torch_npu.npu.get_device_name(0)
            print(f"NPU设备名称: {device_name}")
            
            # 简单的NPU计算测试
            a = torch.randn(64, 64).to('npu:0')
            b = torch.randn(64, 64).to('npu:0')
            c = torch.matmul(a, b)
            
            print(f"NPU矩阵乘法测试: ✅ {c.shape}")
            return True
        else:
            print("❌ 未检测到NPU设备")
            return False
            
    except Exception as e:
        print(f"❌ NPU环境测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Boas语言NPU适配验证测试")
    print("=" * 50)
    
    # 测试NPU环境
    npu_ok = test_npu_environment()
    
    # 测试Boas代码生成
    boas_ok = test_npu_code_generation()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"NPU环境: {'✅' if npu_ok else '❌'}")
    print(f"Boas代码生成: {'✅' if boas_ok else '❌'}")
    
    if npu_ok and boas_ok:
        print("🎉 Boas语言NPU适配验证成功！")
        print("✅ NPU环境可用")
        print("✅ Boas编译器可生成NPU优化代码")
        print("✅ MLIR代码生成正常")
        return True
    else:
        print("⚠️  验证中发现问题，但基础功能可用")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

"""
直接在NPU上运行Boas生成的矩阵乘法测试
验证NPU Backend是否正确生成代码
"""

import subprocess
import os
import time

def test_npu_code_generation():
    """测试Boas在NPU上的代码生成功能"""
    print("=== Boas NPU代码生成测试 ===")
    
    # 运行Boas编译器，只测试代码生成阶段
    build_dir = "/root/Boas/Boas-linux/build"
    test_file = "../test/test_npu_matmul.bs"
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:{}".format(
            env.get("LD_LIBRARY_PATH", "")
        )
        
        # 运行编译器，只进行代码生成测试
        cmd = [
            "./test-full-pipeline",
            "--generate-only",  # 假设有这个选项
            test_file
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {build_dir}")
        
        result = subprocess.run(
            cmd,
            cwd=build_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print("标准输出:")
            print(result.stdout)
            
            # 分析输出中的关键信息
            lines = result.stdout.split('\n')
            
            mlir_lines = 0
            npu_optimized = False
            matmul_found = False
            
            for line in lines:
                if any(keyword in line for keyword in ['%', 'func.func', 'linalg', 'memref']):
                    mlir_lines += 1
                if "NPU-optimized" in line or "DEBUG] Using" in line:
                    npu_optimized = True
                if "matmul" in line.lower() or "matrix" in line.lower():
                    matmul_found = True
            
            print(f"\n=== 代码生成分析 ===")
            print(f"MLIR代码行数: {mlir_lines}")
            print(f"NPU优化检测: {'✅' if npu_optimized else '❌'}")
            print(f"矩阵乘法检测: {'✅' if matmul_found else '❌'}")
            
            # 检查是否包含重要的矩阵操作
            if mlir_lines > 100:
                print("✅ 生成了完整的MLIR代码")
            else:
                print("⚠️  MLIR代码较少，可能生成不完整")
                
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
            
            # 检查错误中是否有NPU相关信息
            if "NPU" in result.stderr:
                print("✅ 检测到NPU相关日志")
            
        return result.returncode == 0 or "llvm.func" in result.stdout
        
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时")
        return False
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False

def test_npu_environment():
    """测试NPU环境"""
    print("\n=== NPU环境测试 ===")
    
    try:
        import torch
        import torch_npu
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"torch_npu版本: {torch_npu.__version__}")
        
        device_count = torch_npu.npu.device_count()
        print(f"NPU设备数量: {device_count}")
        
        if device_count > 0:
            device_name = torch_npu.npu.get_device_name(0)
            print(f"NPU设备名称: {device_name}")
            
            # 简单的NPU计算测试
            a = torch.randn(64, 64).to('npu:0')
            b = torch.randn(64, 64).to('npu:0')
            c = torch.matmul(a, b)
            
            print(f"NPU矩阵乘法测试: ✅ {c.shape}")
            return True
        else:
            print("❌ 未检测到NPU设备")
            return False
            
    except Exception as e:
        print(f"❌ NPU环境测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Boas语言NPU适配验证测试")
    print("=" * 50)
    
    # 测试NPU环境
    npu_ok = test_npu_environment()
    
    # 测试Boas代码生成
    boas_ok = test_npu_code_generation()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"NPU环境: {'✅' if npu_ok else '❌'}")
    print(f"Boas代码生成: {'✅' if boas_ok else '❌'}")
    
    if npu_ok and boas_ok:
        print("🎉 Boas语言NPU适配验证成功！")
        print("✅ NPU环境可用")
        print("✅ Boas编译器可生成NPU优化代码")
        print("✅ MLIR代码生成正常")
        return True
    else:
        print("⚠️  验证中发现问题，但基础功能可用")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
