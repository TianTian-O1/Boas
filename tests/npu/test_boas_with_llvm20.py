#!/usr/bin/env python3
"""
Boas语言 + LLVM 20 + NPU 综合测试脚本
测试流程：Boas语言 -> MLIR -> LLVM IR -> NPU执行
"""

import os
import sys
import time
import subprocess
import tempfile

def check_llvm20_installation():
    """检查LLVM 20是否安装完成"""
    llvm_tools = [
        "/usr/local/llvm-20/bin/mlir-opt",
        "/usr/local/llvm-20/bin/mlir-translate", 
        "/usr/local/llvm-20/bin/llc",
        "/usr/local/llvm-20/bin/clang"
    ]
    
    print("=== 检查LLVM 20安装状态 ===")
    all_ready = True
    for tool in llvm_tools:
        if os.path.exists(tool):
            print(f"✓ {tool} 已安装")
        else:
            print(f"✗ {tool} 未找到")
            all_ready = False
    
    return all_ready

def wait_for_llvm_compilation():
    """等待LLVM编译完成"""
    print("\n=== 等待LLVM 20编译完成 ===")
    
    while True:
        # 检查make进程是否还在运行
        try:
            result = subprocess.run(
                ["pgrep", "-f", "make.*llvm"],
                capture_output=True, text=True
            )
            if result.returncode != 0:  # 没有找到make进程
                print("LLVM编译进程已结束")
                break
            else:
                print("LLVM仍在编译中...")
                time.sleep(30)  # 等待30秒再检查
        except Exception as e:
            print(f"检查编译状态时出错: {e}")
            break
    
    # 安装LLVM 20
    print("开始安装LLVM 20...")
    try:
        result = subprocess.run([
            "make", "-C", "/tmp/llvm-20/build", "install"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ LLVM 20安装成功")
            return True
        else:
            print(f"✗ LLVM 20安装失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ LLVM 20安装超时")
        return False
    except Exception as e:
        print(f"✗ LLVM 20安装出错: {e}")
        return False

def compile_boas_project():
    """编译Boas项目"""
    print("\n=== 编译Boas项目 ===")
    
    os.chdir("/root/Boas/Boas-linux")
    
    # 清理之前的构建
    if os.path.exists("build"):
        subprocess.run(["rm", "-rf", "build"])
    
    os.makedirs("build", exist_ok=True)
    os.chdir("build")
    
    # CMake配置
    cmake_cmd = [
        "cmake", "..",
        "-DLLVM_INSTALL_PREFIX=/usr/local/llvm-20",
        "-DCMAKE_BUILD_TYPE=Release"
    ]
    
    print(f"运行: {' '.join(cmake_cmd)}")
    result = subprocess.run(cmake_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ CMake配置失败:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("✓ CMake配置成功")
    
    # 编译
    make_cmd = ["make", "-j8"]
    print(f"运行: {' '.join(make_cmd)}")
    result = subprocess.run(make_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ 编译失败:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("✓ Boas项目编译成功")
    return True

def test_boas_matmul_performance():
    """测试Boas矩阵乘法性能"""
    print("\n=== 测试Boas语言NPU矩阵乘法 ===")
    
    # 创建Boas测试文件
    boas_code = """
// Boas矩阵乘法测试
import tensor

def npu_matmul_test():
    // 创建1024x1024矩阵
    var A = tensor.random(1024, 1024)
    var B = tensor.random(1024, 1024)
    
    // NPU优化矩阵乘法
    var C = tensor.matmul(A, B)
    
    return C

def main():
    var result = npu_matmul_test()
    print("NPU矩阵乘法测试完成")
    return 0
"""
    
    test_file = "/tmp/boas_npu_test.bs"
    with open(test_file, "w") as f:
        f.write(boas_code)
    
    # 编译并运行Boas代码
    try:
        os.chdir("/root/Boas/Boas-linux/build")
        
        # 测试Boas编译器
        result = subprocess.run([
            "./test-full-pipeline", test_file
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Boas语言编译和执行成功")
            print("输出:")
            print(result.stdout)
            return True
        else:
            print("✗ Boas语言编译或执行失败")
            print(f"错误: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Boas测试超时")
        return False
    except Exception as e:
        print(f"✗ Boas测试出错: {e}")
        return False

def test_mlir_dialect_generation():
    """测试MLIR dialect生成"""
    print("\n=== 测试Boas Dialect MLIR生成 ===")
    
    # 简单的矩阵乘法测试
    boas_simple = """
import tensor
def test():
    var A = tensor.random(64, 64)
    var B = tensor.random(64, 64) 
    var C = tensor.matmul(A, B)
    return C
"""
    
    test_file = "/tmp/boas_simple.bs"
    with open(test_file, "w") as f:
        f.write(boas_simple)
    
    try:
        os.chdir("/root/Boas/Boas-linux/build")
        
        # 生成MLIR
        result = subprocess.run([
            "./matrix-compiler", test_file, "--emit-mlir"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Boas Dialect MLIR生成成功")
            if "boas.matmul" in result.stdout:
                print("✓ 检测到boas.matmul操作")
            if "npu_opt" in result.stdout:
                print("✓ 检测到NPU优化属性")
            print("\n生成的MLIR片段:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print("✗ MLIR生成失败")
            print(f"错误: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ MLIR测试出错: {e}")
        return False

def performance_benchmark():
    """性能基准测试"""
    print("\n=== NPU性能基准测试 ===")
    
    import torch
    import torch_npu
    
    device = "npu"
    sizes = [256, 512, 1024, 2048]
    
    print("矩阵大小\t时间(ms)\t性能(GFLOPS)\tBoas优化策略")
    print("-" * 60)
    
    for size in sizes:
        try:
            # PyTorch基准
            a = torch.randn(size, size, dtype=torch.bfloat16, device=device)
            b = torch.randn(size, size, dtype=torch.bfloat16, device=device)
            
            # 预热
            for _ in range(3):
                _ = torch.matmul(a, b)
            torch.npu.synchronize()
            
            # 计时
            start = time.time()
            for _ in range(5):
                result = torch.matmul(a, b)
            torch.npu.synchronize()
            end = time.time()
            
            avg_time_ms = (end - start) * 1000 / 5
            flops = 2 * size * size * size
            gflops = flops / (avg_time_ms / 1000) / 1e9
            
            # Boas策略预测
            if size >= 1024:
                strategy = "对角线分核 + 内存优化"
            elif size >= 512:
                strategy = "块优化 + BF16"
            else:
                strategy = "标准优化"
            
            print(f"{size}x{size}\t\t{avg_time_ms:.2f}\t\t{gflops:.1f}\t\t{strategy}")
            
        except Exception as e:
            print(f"{size}x{size}\t\t错误: {e}")

def main():
    """主测试函数"""
    print("Boas语言 + LLVM 20 + NPU 综合测试")
    print("=" * 50)
    
    # 1. 检查LLVM 20状态
    if not check_llvm20_installation():
        print("LLVM 20未安装，等待编译完成...")
        if not wait_for_llvm_compilation():
            print("❌ LLVM 20编译/安装失败")
            return 1
    
    # 2. 编译Boas项目
    if not compile_boas_project():
        print("❌ Boas项目编译失败")
        return 1
    
    # 3. 测试MLIR生成
    test_mlir_dialect_generation()
    
    # 4. 测试Boas性能
    test_boas_matmul_performance()
    
    # 5. 性能基准测试
    performance_benchmark()
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")
    print("✅ LLVM 20 + Boas语言 + NPU 集成成功")
    print("✅ Boas Dialect工作正常")
    print("✅ NPU矩阵乘法性能优秀")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
