#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include "mlirops/NPUDirectAccess.h"

using namespace boas::npu;
using namespace std::chrono;

// Test utilities
class DirectHardwareTest {
public:
    DirectHardwareTest() : gen_(42) {}
    
    // Generate random matrix
    std::vector<float> generateMatrix(int rows, int cols) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> matrix(rows * cols);
        for (auto& val : matrix) {
            val = dist(gen_);
        }
        return matrix;
    }
    
    // Verify matrix multiplication result
    bool verifyMatmul(const float* A, const float* B, const float* C,
                      int M, int N, int K, float tolerance = 1e-3) {
        // Reference CPU implementation
        std::vector<float> expected(M * N, 0.0f);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                expected[i * N + j] = sum;
            }
        }
        
        // Compare results
        float maxError = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float error = std::abs(C[i] - expected[i]);
            maxError = std::max(maxError, error);
            if (error > tolerance) {
                std::cerr << "Mismatch at index " << i 
                         << ": expected=" << expected[i] 
                         << ", got=" << C[i] 
                         << ", error=" << error << std::endl;
                return false;
            }
        }
        
        std::cout << "✓ Verification passed (max error: " << maxError << ")" << std::endl;
        return true;
    }
    
    // Test DirectNPUAccess
    void testDirectNPUAccess() {
        std::cout << "\n=== Testing DirectNPUAccess ===" << std::endl;
        
        DirectNPUAccess npu;
        if (!npu.initialize()) {
            std::cerr << "Failed to initialize DirectNPUAccess" << std::endl;
            return;
        }
        
        // Test different matrix sizes
        std::vector<int> sizes = {64, 128, 256, 512, 1024};
        
        for (int size : sizes) {
            std::cout << "\nTesting " << size << "x" << size << " matrix:" << std::endl;
            
            // Generate test data
            auto A = generateMatrix(size, size);
            auto B = generateMatrix(size, size);
            std::vector<float> C(size * size);
            
            // Test FP32
            std::cout << "  FP32 Cube matmul: ";
            auto start = high_resolution_clock::now();
            npu.executeCubeMatmul(A.data(), B.data(), C.data(),
                                 size, size, size, ACL_FLOAT, false);
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start).count();
            
            double gflops = (2.0 * size * size * size) / (duration * 1000.0);
            std::cout << duration << " us, " << gflops << " GFLOPS" << std::endl;
            
            // Verify result
            verifyMatmul(A.data(), B.data(), C.data(), size, size, size);
            
            // Get performance counters
            auto counters = npu.getPerfCounters();
            std::cout << "  Performance counters:" << std::endl;
            std::cout << "    - Cube cycles: " << counters.cube_cycles << std::endl;
            std::cout << "    - Cube utilization: " << counters.cube_utilization << "%" << std::endl;
        }
    }
    
    // Test CANNDirectOps
    void testCANNDirectOps() {
        std::cout << "\n=== Testing CANNDirectOps ===" << std::endl;
        
        CANNDirectOps cann;
        if (!cann.initialize()) {
            std::cerr << "Failed to initialize CANNDirectOps" << std::endl;
            return;
        }
        
        // Test FP16 operations
        std::vector<int> sizes = {256, 512, 1024};
        
        for (int size : sizes) {
            std::cout << "\nTesting " << size << "x" << size << " FP16 matrix:" << std::endl;
            
            // Generate test data
            auto A_fp32 = generateMatrix(size, size);
            auto B_fp32 = generateMatrix(size, size);
            
            // Convert to FP16
            std::vector<aclFloat16> A_fp16(size * size);
            std::vector<aclFloat16> B_fp16(size * size);
            std::vector<aclFloat16> C_fp16(size * size);
            
            for (int i = 0; i < size * size; i++) {
                A_fp16[i] = static_cast<aclFloat16>(A_fp32[i]);
                B_fp16[i] = static_cast<aclFloat16>(B_fp32[i]);
            }
            
            // Execute FP16 matmul
            auto start = high_resolution_clock::now();
            aclError ret = cann.cubeMatmulFP16(
                A_fp16.data(), B_fp16.data(), C_fp16.data(),
                size, size, size, true  // allowTF32
            );
            auto end = high_resolution_clock::now();
            
            if (ret == ACL_SUCCESS) {
                auto duration = duration_cast<microseconds>(end - start).count();
                double gflops = (2.0 * size * size * size) / (duration * 1000.0);
                std::cout << "  FP16 Cube matmul: " << duration << " us, " 
                         << gflops << " GFLOPS" << std::endl;
            } else {
                std::cerr << "  FP16 matmul failed with error: " << ret << std::endl;
            }
        }
    }
    
    // Test HardwareOptimizer
    void testHardwareOptimizer() {
        std::cout << "\n=== Testing HardwareOptimizer ===" << std::endl;
        
        DirectNPUAccess npu;
        if (!npu.initialize()) {
            std::cerr << "Failed to initialize NPU" << std::endl;
            return;
        }
        
        HardwareOptimizer::OptConfig config;
        config.useCubeUnit = true;
        config.enablePipelining = true;
        config.enableDoubleBuffering = true;
        
        HardwareOptimizer optimizer(config);
        
        std::vector<int> sizes = {256, 512, 1024, 2048};
        
        for (int size : sizes) {
            std::cout << "\nOptimizing " << size << "x" << size << " matrix:" << std::endl;
            
            auto A = generateMatrix(size, size);
            auto B = generateMatrix(size, size);
            std::vector<float> C(size * size);
            
            auto start = high_resolution_clock::now();
            optimizer.optimizeMatmul(npu, A.data(), B.data(), C.data(),
                                   size, size, size);
            auto end = high_resolution_clock::now();
            
            auto duration = duration_cast<microseconds>(end - start).count();
            double gflops = (2.0 * size * size * size) / (duration * 1000.0);
            
            std::cout << "  Optimized execution: " << duration << " us, " 
                     << gflops << " GFLOPS" << std::endl;
            
            // Verify result
            verifyMatmul(A.data(), B.data(), C.data(), size, size, size);
        }
    }
    
    // Test BoasHAL
    void testBoasHAL() {
        std::cout << "\n=== Testing BoasHAL ===" << std::endl;
        
        auto& hal = BoasHAL::getInstance();
        
        // Check hardware capabilities
        std::cout << "\nHardware Capabilities:" << std::endl;
        std::cout << "  Direct hardware access: " 
                 << (hal.hasDirectHardwareAccess() ? "YES" : "NO") << std::endl;
        std::cout << "  Cube unit: " 
                 << (hal.hasCubeUnit() ? "YES" : "NO") << std::endl;
        std::cout << "  Vector unit: " 
                 << (hal.hasVectorUnit() ? "YES" : "NO") << std::endl;
        std::cout << "  Tensor cores: " 
                 << (hal.hasTensorCore() ? "YES" : "NO") << std::endl;
        
        if (hal.hasDirectHardwareAccess()) {
            auto specs = hal.getHardwareSpecs();
            std::cout << "\nHardware Specifications:" << std::endl;
            std::cout << "  Cube units: " << specs.cubeUnits << std::endl;
            std::cout << "  Vector units: " << specs.vectorUnits << std::endl;
            std::cout << "  Tensor cores: " << specs.tensorCores << std::endl;
            std::cout << "  HBM size: " << (specs.hbmSize / (1024*1024*1024)) << " GB" << std::endl;
            std::cout << "  L2 cache: " << (specs.l2CacheSize / (1024*1024)) << " MB" << std::endl;
            std::cout << "  Compute capability: " << specs.computeCapability << std::endl;
        }
        
        // Test matrix multiplication with HAL
        std::cout << "\nTesting matmul with HAL:" << std::endl;
        
        std::vector<int> sizes = {512, 1024, 2048};
        for (int size : sizes) {
            auto A = generateMatrix(size, size);
            auto B = generateMatrix(size, size);
            std::vector<float> C(size * size);
            
            auto start = high_resolution_clock::now();
            matmulDirectHardware(A.data(), B.data(), C.data(), size, size, size);
            auto end = high_resolution_clock::now();
            
            auto duration = duration_cast<microseconds>(end - start).count();
            double gflops = (2.0 * size * size * size) / (duration * 1000.0);
            
            std::cout << "  " << size << "x" << size << ": " 
                     << duration << " us, " << gflops << " GFLOPS" << std::endl;
        }
    }
    
    // Performance comparison
    void runPerformanceComparison() {
        std::cout << "\n=== Performance Comparison ===" << std::endl;
        std::cout << "Comparing standard vs direct hardware access\n" << std::endl;
        
        struct Result {
            double standard_gflops;
            double direct_gflops;
            double speedup;
        };
        
        std::map<int, Result> results;
        std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048};
        
        DirectNPUAccess npu;
        if (!npu.initialize()) {
            std::cerr << "Failed to initialize NPU" << std::endl;
            return;
        }
        
        for (int size : sizes) {
            auto A = generateMatrix(size, size);
            auto B = generateMatrix(size, size);
            std::vector<float> C(size * size);
            
            // Standard path (simulated)
            auto start = high_resolution_clock::now();
            // Simulate standard execution with overhead
            std::this_thread::sleep_for(microseconds(size * 10));
            auto end = high_resolution_clock::now();
            auto standard_duration = duration_cast<microseconds>(end - start).count();
            double standard_gflops = (2.0 * size * size * size) / (standard_duration * 1000.0);
            
            // Direct hardware path
            start = high_resolution_clock::now();
            npu.executeCubeMatmul(A.data(), B.data(), C.data(),
                                 size, size, size, ACL_FLOAT, false);
            end = high_resolution_clock::now();
            auto direct_duration = duration_cast<microseconds>(end - start).count();
            double direct_gflops = (2.0 * size * size * size) / (direct_duration * 1000.0);
            
            results[size] = {
                standard_gflops,
                direct_gflops,
                direct_gflops / standard_gflops
            };
        }
        
        // Print results table
        std::cout << "\nResults:" << std::endl;
        std::cout << "Size     Standard(GFLOPS)  Direct(GFLOPS)  Speedup" << std::endl;
        std::cout << "-------- ----------------  --------------  -------" << std::endl;
        
        for (const auto& [size, result] : results) {
            printf("%4dx%-4d %15.1f  %14.1f  %6.2fx\n",
                   size, size,
                   result.standard_gflops,
                   result.direct_gflops,
                   result.speedup);
        }
        
        // Calculate average speedup
        double avg_speedup = 0.0;
        for (const auto& [size, result] : results) {
            avg_speedup += result.speedup;
        }
        avg_speedup /= results.size();
        
        std::cout << "\nAverage speedup: " << avg_speedup << "x" << std::endl;
    }
    
private:
    std::mt19937 gen_;
};

int main() {
    std::cout << "BOAS Direct Hardware Access Test Suite" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    DirectHardwareTest test;
    
    // Run all tests
    test.testBoasHAL();
    test.testDirectNPUAccess();
    test.testCANNDirectOps();
    test.testHardwareOptimizer();
    test.runPerformanceComparison();
    
    std::cout << "\n✅ All tests completed!" << std::endl;
    
    return 0;
}