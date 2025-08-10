#!/usr/bin/env python3
"""
Test 2x2 matrix multiplication to verify correctness
"""

import numpy as np
import torch
import torch_npu
import time

def test_2x2_matrix():
    """Test 2x2 matrix multiplication on NPU"""
    print("="*60)
    print("Testing 2x2 Matrix Multiplication on NPU")
    print("="*60)
    
    # Create specific 2x2 matrices for testing
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]], dtype=np.float32)
    
    B = np.array([[5.0, 6.0],
                  [7.0, 8.0]], dtype=np.float32)
    
    # Expected result (calculated manually)
    # C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
    # C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
    # C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
    # C[1,1] = 3*6 + 4*8 = 18 + 32 = 50
    expected = np.array([[19.0, 22.0],
                        [43.0, 50.0]], dtype=np.float32)
    
    print("\nInput Matrix A:")
    print(A)
    print("\nInput Matrix B:")
    print(B)
    print("\nExpected Result (A @ B):")
    print(expected)
    
    # Test on CPU first (numpy)
    print("\n" + "-"*40)
    print("Testing with NumPy (CPU):")
    cpu_result = np.matmul(A, B)
    print("Result:")
    print(cpu_result)
    print(f"Correct: {np.allclose(cpu_result, expected)}")
    
    # Test on NPU if available
    if torch_npu.npu.is_available():
        print("\n" + "-"*40)
        print("Testing with PyTorch on NPU:")
        
        # Convert to PyTorch tensors and move to NPU
        A_npu = torch.from_numpy(A).npu()
        B_npu = torch.from_numpy(B).npu()
        
        # Perform matrix multiplication
        C_npu = torch.matmul(A_npu, B_npu)
        
        # Move result back to CPU
        C_cpu = C_npu.cpu().numpy()
        
        print("Result:")
        print(C_cpu)
        print(f"Correct: {np.allclose(C_cpu, expected, rtol=1e-5)}")
        
        # Check individual elements
        print("\nElement-wise comparison:")
        for i in range(2):
            for j in range(2):
                diff = abs(C_cpu[i, j] - expected[i, j])
                print(f"  C[{i},{j}]: {C_cpu[i, j]:.6f} (expected: {expected[i, j]:.6f}, diff: {diff:.6e})")
        
        # Test with random matrices
        print("\n" + "-"*40)
        print("Testing with random 2x2 matrices:")
        
        for test_num in range(3):
            print(f"\nTest {test_num + 1}:")
            
            # Generate random 2x2 matrices
            A_rand = np.random.randn(2, 2).astype(np.float32)
            B_rand = np.random.randn(2, 2).astype(np.float32)
            
            # CPU reference
            expected_rand = np.matmul(A_rand, B_rand)
            
            # NPU computation
            A_rand_npu = torch.from_numpy(A_rand).npu()
            B_rand_npu = torch.from_numpy(B_rand).npu()
            C_rand_npu = torch.matmul(A_rand_npu, B_rand_npu)
            C_rand_cpu = C_rand_npu.cpu().numpy()
            
            # Compare
            max_diff = np.max(np.abs(C_rand_cpu - expected_rand))
            is_correct = np.allclose(C_rand_cpu, expected_rand, rtol=1e-5)
            
            print(f"  Max difference: {max_diff:.6e}")
            print(f"  Correct: {is_correct}")
            
            if not is_correct:
                print("  WARNING: Result mismatch!")
                print(f"  NPU result:\n{C_rand_cpu}")
                print(f"  Expected:\n{expected_rand}")
        
        # Performance test for 2x2
        print("\n" + "-"*40)
        print("Performance test (2x2 matrix, 1000 iterations):")
        
        iterations = 1000
        
        # Warmup
        for _ in range(100):
            _ = torch.matmul(A_npu, B_npu)
        torch_npu.npu.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            C_npu = torch.matmul(A_npu, B_npu)
        torch_npu.npu.synchronize()
        end = time.perf_counter()
        
        total_time = (end - start) * 1000  # ms
        avg_time = total_time / iterations  # ms per operation
        
        # Calculate FLOPS (2*2*2*2 = 16 operations per matmul)
        flops = 16 * iterations / (total_time / 1000)  # FLOPS
        
        print(f"  Total time: {total_time:.3f} ms")
        print(f"  Average time per matmul: {avg_time:.6f} ms")
        print(f"  Performance: {flops:.0f} FLOPS ({flops/1e9:.6f} GFLOPS)")
        
    else:
        print("\nNPU not available, skipping NPU tests")
    
    # Test larger matrices for comparison
    print("\n" + "="*60)
    print("Testing various small matrix sizes:")
    print("="*60)
    
    sizes = [2, 4, 8, 16, 32]
    
    for size in sizes:
        print(f"\n{size}x{size} matrix:")
        
        # Create random matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # CPU reference
        cpu_result = np.matmul(A, B)
        
        if torch_npu.npu.is_available():
            # NPU computation
            A_npu = torch.from_numpy(A).npu()
            B_npu = torch.from_numpy(B).npu()
            
            # Warmup
            for _ in range(10):
                _ = torch.matmul(A_npu, B_npu)
            torch_npu.npu.synchronize()
            
            # Time single operation
            torch_npu.npu.synchronize()
            start = time.perf_counter()
            C_npu = torch.matmul(A_npu, B_npu)
            torch_npu.npu.synchronize()
            end = time.perf_counter()
            
            npu_result = C_npu.cpu().numpy()
            
            # Verify correctness
            max_diff = np.max(np.abs(npu_result - cpu_result))
            is_correct = np.allclose(npu_result, cpu_result, rtol=1e-4)
            
            # Calculate performance
            time_ms = (end - start) * 1000
            flops = 2 * size**3
            gflops = flops / (time_ms * 1e6)
            
            print(f"  Time: {time_ms:.6f} ms")
            print(f"  Performance: {gflops:.6f} GFLOPS")
            print(f"  Max error: {max_diff:.6e}")
            print(f"  Correct: {is_correct}")
            
            if not is_correct:
                print(f"  WARNING: Large error detected!")

def test_boas_simulation():
    """Simulate BOAS optimization for 2x2 matrix"""
    print("\n" + "="*60)
    print("BOAS Optimization Simulation for 2x2 Matrix")
    print("="*60)
    
    # Test matrices
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]], dtype=np.float32)
    
    B = np.array([[5.0, 6.0],
                  [7.0, 8.0]], dtype=np.float32)
    
    print("\nStandard computation:")
    print("-"*40)
    
    # Standard matmul
    C_standard = np.matmul(A, B)
    print(f"Result:\n{C_standard}")
    
    print("\nBOAS optimized computation (simulated):")
    print("-"*40)
    
    # Simulate BOAS optimizations
    # For 2x2, BOAS would:
    # 1. Use vector instructions
    # 2. Unroll loops completely
    # 3. Keep all values in registers
    
    # Unrolled computation (what BOAS would generate)
    c00 = A[0,0] * B[0,0] + A[0,1] * B[1,0]
    c01 = A[0,0] * B[0,1] + A[0,1] * B[1,1]
    c10 = A[1,0] * B[0,0] + A[1,1] * B[1,0]
    c11 = A[1,0] * B[0,1] + A[1,1] * B[1,1]
    
    C_boas = np.array([[c00, c01],
                       [c10, c11]], dtype=np.float32)
    
    print(f"Result:\n{C_boas}")
    print(f"Matches standard: {np.allclose(C_boas, C_standard)}")
    
    print("\nOptimization details for 2x2:")
    print("  • Loop fully unrolled (no branches)")
    print("  • All values in registers")
    print("  • Vector instructions for parallel multiply-add")
    print("  • No memory access during computation")
    print("  • Theoretical peak: ~16 operations in 1 cycle")

if __name__ == "__main__":
    print("BOAS 2x2 Matrix Multiplication Test")
    print("="*60)
    
    # Run tests
    test_2x2_matrix()
    test_boas_simulation()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)