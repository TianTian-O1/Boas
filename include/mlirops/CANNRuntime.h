#pragma once

#ifdef HAVE_CANN_SUPPORT
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnn/aclnn_base.h"
#endif

#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace matrix {

/// CANN运行时管理类
/// 负责ACL初始化、设备管理、内存管理等
class CANNRuntime {
public:
    /// 单例模式获取运行时实例
    static CANNRuntime& getInstance();
    
    /// 初始化CANN运行时
    bool initialize();
    
    /// 清理CANN运行时
    void finalize();
    
    /// 检查CANN是否可用
    bool isAvailable() const;
    
    /// 获取NPU设备数量
    uint32_t getDeviceCount() const;
    
    /// 设置当前设备
    bool setDevice(uint32_t deviceId);
    
    /// 获取当前设备ID
    uint32_t getCurrentDevice() const;
    
    /// 获取设备属性
    std::string getDeviceProperties(uint32_t deviceId) const;
    
    /// NPU内存分配
    void* allocateMemory(size_t size);
    
    /// NPU内存释放
    bool freeMemory(void* ptr);
    
    /// 主机到NPU内存拷贝
    bool copyHostToDevice(void* dst, const void* src, size_t size);
    
    /// NPU到主机内存拷贝  
    bool copyDeviceToHost(void* dst, const void* src, size_t size);
    
    /// 执行NPU矩阵乘法
    bool executeMatmul(
        const float* a, const float* b, float* c,
        int64_t m, int64_t k, int64_t n,
        bool transposeA = false, bool transposeB = false
    );
    
    /// 同步NPU执行
    bool synchronize();
    
    /// 获取最后的错误信息
    std::string getLastError() const;

private:
    CANNRuntime() = default;
    ~CANNRuntime();
    
    // 禁用拷贝构造
    CANNRuntime(const CANNRuntime&) = delete;
    CANNRuntime& operator=(const CANNRuntime&) = delete;
    
#ifdef HAVE_CANN_SUPPORT
    /// ACL错误码转换为字符串
    std::string aclErrorToString(aclError error) const;
    
    /// 检查ACL错误
    bool checkAclError(aclError error, const std::string& operation) const;
#endif
    
    bool initialized_ = false;
    uint32_t deviceCount_ = 0;
    uint32_t currentDevice_ = 0;
    mutable std::string lastError_;
    
#ifdef HAVE_CANN_SUPPORT
    aclrtStream stream_ = nullptr;
    aclrtContext context_ = nullptr;
#endif
};

/// RAII风格的NPU内存管理器
class NPUMemoryBuffer {
public:
    explicit NPUMemoryBuffer(size_t size);
    ~NPUMemoryBuffer();
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    bool copyFromHost(const void* src, size_t copySize = 0);
    bool copyToHost(void* dst, size_t copySize = 0) const;
    
private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

} // namespace matrix
