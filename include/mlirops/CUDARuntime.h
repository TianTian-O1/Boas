#pragma once

#ifdef HAVE_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace matrix {

/// CUDA运行时管理类
/// 负责CUDA初始化、设备管理、内存管理等
class CUDARuntime {
public:
    /// 单例模式获取运行时实例
    static CUDARuntime& getInstance();

    /// 初始化CUDA运行时
    bool initialize();

    /// 清理CUDA运行时
    void finalize();

    /// 检查CUDA是否可用
    bool isAvailable() const;

    /// 获取GPU设备数量
    int getDeviceCount() const;

    /// 设置当前设备
    bool setDevice(int deviceId);

    /// 获取当前设备ID
    int getCurrentDevice() const;

    /// 获取设备属性
    std::string getDeviceProperties(int deviceId) const;

    /// GPU内存分配
    void* allocateMemory(size_t size);

    /// GPU内存释放
    bool freeMemory(void* ptr);

    /// 主机到GPU内存拷贝
    bool copyHostToDevice(void* dst, const void* src, size_t size);

    /// GPU到主机内存拷贝
    bool copyDeviceToHost(void* dst, const void* src, size_t size);

    /// 执行GPU矩阵乘法（使用cuBLAS）
    bool executeMatmul(
        const float* a, const float* b, float* c,
        int m, int k, int n,
        bool transposeA = false, bool transposeB = false
    );

    /// 执行自定义CUDA kernel矩阵乘法
    bool executeCustomMatmul(
        const float* a, const float* b, float* c,
        int m, int k, int n
    );

    /// 同步GPU执行
    bool synchronize();

    /// 获取最后的错误信息
    std::string getLastError() const;

    /// 获取cuBLAS句柄
#ifdef HAVE_CUDA_SUPPORT
    cublasHandle_t getCublasHandle() const { return cublasHandle_; }
#endif

private:
    CUDARuntime() = default;
    ~CUDARuntime();

    // 禁用拷贝构造
    CUDARuntime(const CUDARuntime&) = delete;
    CUDARuntime& operator=(const CUDARuntime&) = delete;

#ifdef HAVE_CUDA_SUPPORT
    /// CUDA错误码转换为字符串
    std::string cudaErrorToString(cudaError_t error) const;

    /// cuBLAS错误码转换为字符串
    std::string cublasErrorToString(cublasStatus_t status) const;

    /// 检查CUDA错误
    bool checkCudaError(cudaError_t error, const std::string& operation) const;

    /// 检查cuBLAS错误
    bool checkCublasError(cublasStatus_t status, const std::string& operation) const;
#endif

    bool initialized_ = false;
    int deviceCount_ = 0;
    int currentDevice_ = 0;
    mutable std::string lastError_;

#ifdef HAVE_CUDA_SUPPORT
    cublasHandle_t cublasHandle_ = nullptr;
#endif
};

/// RAII风格的GPU内存管理器
class GPUMemoryBuffer {
public:
    explicit GPUMemoryBuffer(size_t size);
    ~GPUMemoryBuffer();

    void* get() const { return ptr_; }
    size_t size() const { return size_; }

    bool copyFromHost(const void* src, size_t copySize = 0);
    bool copyToHost(void* dst, size_t copySize = 0) const;

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

} // namespace matrix
