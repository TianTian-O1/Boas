#include "mlirops/CUDARuntime.h"
#include "mlirops/GPUBackend.h"
#include "mlirops/DeviceManager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace matrix;

// 验证矩阵乘法结果的正确性
bool verifyMatmul(const float* A, const float* B, const float* C,
                  int M, int N, int K, float tolerance = 1e-4) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += A[i * K + k] * B[k * N + j];
            }
            float actual = C[i * N + j];
            if (std::abs(actual - expected) > tolerance) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << "expected " << expected << ", got " << actual << std::endl;
                return false;
            }
        }
    }
    return true;
}

void testDeviceManager() {
    std::cout << "\n=== Testing DeviceManager ===" << std::endl;

    auto& deviceMgr = DeviceManager::getInstance();

    if (!deviceMgr.initialize()) {
        std::cerr << "Failed to initialize DeviceManager" << std::endl;
        return;
    }

    deviceMgr.printAvailableDevices();

    auto currentDevice = deviceMgr.getCurrentDevice();
    std::cout << "\nCurrent device: " << deviceMgr.deviceTypeToString(currentDevice.type)
              << " " << currentDevice.deviceId << std::endl;
}

void testCUDARuntime() {
    std::cout << "\n=== Testing CUDA Runtime ===" << std::endl;

    auto& cudaRuntime = CUDARuntime::getInstance();

    if (!cudaRuntime.initialize()) {
        std::cerr << "CUDA not available: " << cudaRuntime.getLastError() << std::endl;
        return;
    }

    std::cout << "CUDA devices: " << cudaRuntime.getDeviceCount() << std::endl;
    for (int i = 0; i < cudaRuntime.getDeviceCount(); i++) {
        std::cout << "  Device " << i << ": "
                  << cudaRuntime.getDeviceProperties(i) << std::endl;
    }
}

void testSimpleMatmul() {
    std::cout << "\n=== Testing Simple Matmul (4x4) ===" << std::endl;

    auto& cudaRuntime = CUDARuntime::getInstance();

    if (!cudaRuntime.isAvailable()) {
        std::cerr << "CUDA not available" << std::endl;
        return;
    }

    const int M = 4, N = 4, K = 4;

    // 主机端数据
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    // GPU内存分配
    GPUMemoryBuffer devA(M * K * sizeof(float));
    GPUMemoryBuffer devB(K * N * sizeof(float));
    GPUMemoryBuffer devC(M * N * sizeof(float));

    // 拷贝到GPU
    devA.copyFromHost(A.data());
    devB.copyFromHost(B.data());

    // 执行矩阵乘法
    bool success = cudaRuntime.executeMatmul(
        (float*)devA.get(), (float*)devB.get(), (float*)devC.get(),
        M, K, N
    );

    if (!success) {
        std::cerr << "Matmul failed: " << cudaRuntime.getLastError() << std::endl;
        return;
    }

    // 同步
    cudaRuntime.synchronize();

    // 拷贝回主机
    devC.copyToHost(C.data());

    // 验证结果
    std::cout << "Result C[0][0] = " << C[0] << " (expected: " << K * 2.0f << ")" << std::endl;

    if (std::abs(C[0] - K * 2.0f) < 1e-4) {
        std::cout << "✅ Simple matmul test passed!" << std::endl;
    } else {
        std::cout << "❌ Simple matmul test failed!" << std::endl;
    }
}

void benchmarkMatmul(int M, int N, int K) {
    std::cout << "\n=== Benchmarking Matmul " << M << "x" << K << " x " << K << "x" << N << " ===" << std::endl;

    auto& cudaRuntime = CUDARuntime::getInstance();

    if (!cudaRuntime.isAvailable()) {
        std::cerr << "CUDA not available" << std::endl;
        return;
    }

    // 主机端数据
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    // 随机初始化
    for (auto& val : A) val = static_cast<float>(rand()) / RAND_MAX;
    for (auto& val : B) val = static_cast<float>(rand()) / RAND_MAX;

    // GPU内存分配
    GPUMemoryBuffer devA(M * K * sizeof(float));
    GPUMemoryBuffer devB(K * N * sizeof(float));
    GPUMemoryBuffer devC(M * N * sizeof(float));

    // 拷贝到GPU
    devA.copyFromHost(A.data());
    devB.copyFromHost(B.data());

    // 预热
    cudaRuntime.executeMatmul(
        (float*)devA.get(), (float*)devB.get(), (float*)devC.get(),
        M, K, N
    );
    cudaRuntime.synchronize();

    // 基准测试
    const int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        cudaRuntime.executeMatmul(
            (float*)devA.get(), (float*)devB.get(), (float*)devC.get(),
            M, K, N
        );
    }
    cudaRuntime.synchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgTime = duration.count() / (double)iterations / 1000.0; // ms
    double gflops = (2.0 * M * N * K) / (avgTime / 1000.0) / 1e9;

    std::cout << "Average time: " << avgTime << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // 验证结果
    devC.copyToHost(C.data());

    // 小矩阵验证正确性
    if (M <= 128 && N <= 128 && K <= 128) {
        if (verifyMatmul(A.data(), B.data(), C.data(), M, N, K)) {
            std::cout << "✅ Result verified!" << std::endl;
        } else {
            std::cout << "❌ Result verification failed!" << std::endl;
        }
    }
}

int main() {
    std::cout << "BOAS GPU Backend Test Suite" << std::endl;
    std::cout << "============================\n" << std::endl;

    // 测试设备管理器
    testDeviceManager();

    // 测试CUDA运行时
    testCUDARuntime();

    // 测试简单矩阵乘法
    testSimpleMatmul();

    // 基准测试不同大小的矩阵
    benchmarkMatmul(128, 128, 128);
    benchmarkMatmul(512, 512, 512);
    benchmarkMatmul(1024, 1024, 1024);
    benchmarkMatmul(2048, 2048, 2048);

    std::cout << "\n=============================" << std::endl;
    std::cout << "All tests completed!" << std::endl;

    return 0;
}
