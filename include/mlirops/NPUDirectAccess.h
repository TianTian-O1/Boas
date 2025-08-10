#ifndef BOAS_NPU_DIRECT_ACCESS_H
#define BOAS_NPU_DIRECT_ACCESS_H

#include <cstdint>
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/aclnn.h"

namespace boas {
namespace npu {

/**
 * BOAS Direct NPU Hardware Access Layer
 * Provides direct access to Ascend NPU hardware features
 */
class DirectNPUAccess {
public:
    // NPU Hardware Components
    enum class ComputeUnit {
        CUBE,       // Matrix computation unit
        VECTOR,     // Vector computation unit  
        SCALAR,     // Scalar computation unit
        DMA         // Direct memory access unit
    };
    
    // Memory Types
    enum class MemoryType {
        HBM,        // High Bandwidth Memory
        L2_CACHE,   // L2 Cache
        L1_BUFFER,  // L1 Buffer
        UB,         // Unified Buffer
        L0C_BUFFER  // L0C Buffer for Cube unit
    };

    DirectNPUAccess();
    ~DirectNPUAccess();
    
    // Initialize direct hardware access
    bool initialize();
    
    // Direct Cube unit operations
    void executeCubeMatmul(
        const void* A, const void* B, void* C,
        int M, int N, int K,
        aclDataType dtype,
        bool useTensorCore = true
    );
    
    // Direct Vector unit operations
    void executeVectorOp(
        const void* input, void* output,
        int size, 
        const char* operation // "relu", "sigmoid", etc
    );
    
    // Direct memory operations
    void* allocateHBM(size_t size);
    void* allocateL1Buffer(size_t size);
    void copyToDevice(void* dst, const void* src, size_t size);
    void copyFromDevice(void* dst, const void* src, size_t size);
    
    // Hardware intrinsics
    void setComputeUnit(ComputeUnit unit);
    void setPrecision(aclDataType dtype);
    void enableFusion(bool enable);
    void setTileSize(int M, int N, int K);
    
    // Performance monitoring
    struct PerfCounters {
        uint64_t cube_cycles;
        uint64_t vector_cycles;
        uint64_t memory_bandwidth;
        double cube_utilization;
        double vector_utilization;
    };
    
    PerfCounters getPerfCounters();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * CANN Low-level API Integration
 * Direct access to CANN runtime and operators
 */
class CANNDirectOps {
public:
    CANNDirectOps();
    ~CANNDirectOps();
    
    // Initialize CANN environment
    bool initialize();
    
    // Direct operator calls
    aclError matmulDirect(
        const aclTensorDesc* descA,
        const aclDataBuffer* A,
        const aclTensorDesc* descB, 
        const aclDataBuffer* B,
        const aclTensorDesc* descC,
        aclDataBuffer* C,
        const aclopAttr* attr = nullptr
    );
    
    // Cube unit specific operations
    aclError cubeMatmulInt8(
        const void* A, const void* B, void* C,
        int M, int N, int K,
        int8_t* workspace,
        size_t workspaceSize
    );
    
    aclError cubeMatmulFP16(
        const aclFloat16* A, 
        const aclFloat16* B,
        aclFloat16* C,
        int M, int N, int K,
        bool allowTF32 = false
    );
    
    // Vector unit specific operations
    aclError vectorAdd(
        const void* A, const void* B, void* C,
        int size, aclDataType dtype
    );
    
    aclError vectorActivation(
        const void* input, void* output,
        int size, 
        aclnnActivationMode mode
    );
    
    // Memory management
    aclError allocateDeviceMemory(void** ptr, size_t size);
    aclError freeDeviceMemory(void* ptr);
    aclError memcpyAsync(
        void* dst, const void* src, 
        size_t size,
        aclrtMemcpyKind kind,
        aclrtStream stream
    );
    
    // Stream management
    aclrtStream createStream();
    aclError destroyStream(aclrtStream stream);
    aclError synchronizeStream(aclrtStream stream);
    
private:
    aclrtContext context_;
    aclrtStream stream_;
    bool initialized_;
};

/**
 * Hardware Optimization Strategies
 * Implements hardware-specific optimizations
 */
class HardwareOptimizer {
public:
    struct OptConfig {
        bool useCubeUnit = true;
        bool useVectorUnit = true;
        bool enablePipelining = true;
        bool enableDoubleBuffering = true;
        bool enableAsyncExecution = true;
        int cubeBlockSize = 16;  // Cube unit block size
        int vectorWidth = 128;    // Vector unit width
    };
    
    HardwareOptimizer(const OptConfig& config = {});
    
    // Optimize matrix multiplication for hardware
    void optimizeMatmul(
        DirectNPUAccess& npu,
        const float* A, const float* B, float* C,
        int M, int N, int K
    );
    
    // Pipeline multiple operations
    void pipelineOperations(
        DirectNPUAccess& npu,
        const std::vector<std::function<void()>>& operations
    );
    
    // Double buffering for hiding memory latency
    template<typename T>
    void doubleBufferedCompute(
        DirectNPUAccess& npu,
        const T* input, T* output,
        size_t batchSize, size_t elementSize,
        std::function<void(const T*, T*, size_t)> computeFunc
    );
    
    // Fuse multiple operations
    void fuseOperations(
        CANNDirectOps& cann,
        const std::vector<aclTensorDesc*>& inputs,
        const std::vector<aclDataBuffer*>& buffers,
        aclTensorDesc* output,
        aclDataBuffer* outputBuffer,
        const std::string& fusionPattern
    );
    
private:
    OptConfig config_;
    
    // Hardware-specific optimization methods
    void optimizeForCubeUnit(int& M, int& N, int& K);
    void optimizeMemoryLayout(void* data, size_t size);
    void prefetchData(const void* data, size_t size);
};

/**
 * BOAS Hardware Abstraction Layer (HAL)
 * Provides unified interface for hardware access
 */
class BoasHAL {
public:
    static BoasHAL& getInstance();
    
    // Check hardware capabilities
    bool hasDirectHardwareAccess() const;
    bool hasCubeUnit() const;
    bool hasVectorUnit() const;
    bool hasTensorCore() const;
    
    // Get hardware specifications
    struct HardwareSpecs {
        int cubeUnits;
        int vectorUnits;
        int tensorCores;
        size_t hbmSize;
        size_t l2CacheSize;
        size_t l1BufferSize;
        int maxThreads;
        int computeCapability;
    };
    
    HardwareSpecs getHardwareSpecs() const;
    
    // Execute with optimal hardware selection
    template<typename Func>
    void executeOptimal(Func&& func) {
        if (hasDirectHardwareAccess()) {
            // Use direct hardware path
            directNPU_.initialize();
            func(directNPU_);
        } else {
            // Fall back to standard path
            executeStandard(std::forward<Func>(func));
        }
    }
    
    // Get direct access interfaces
    DirectNPUAccess& getDirectNPU() { return directNPU_; }
    CANNDirectOps& getCANNOps() { return cannOps_; }
    HardwareOptimizer& getOptimizer() { return optimizer_; }
    
private:
    BoasHAL();
    ~BoasHAL();
    
    DirectNPUAccess directNPU_;
    CANNDirectOps cannOps_;
    HardwareOptimizer optimizer_;
    
    bool checkHardwareAccess();
    
    template<typename Func>
    void executeStandard(Func&& func);
};

/**
 * Example usage for matrix multiplication with direct hardware access
 */
inline void matmulDirectHardware(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    
    auto& hal = BoasHAL::getInstance();
    
    if (hal.hasDirectHardwareAccess()) {
        // Direct hardware path
        auto& npu = hal.getDirectNPU();
        auto& optimizer = hal.getOptimizer();
        
        // Use hardware optimizer
        optimizer.optimizeMatmul(npu, A, B, C, M, N, K);
        
    } else {
        // Fallback to CANN ops
        auto& cann = hal.getCANNOps();
        
        // Allocate device memory
        void *dA, *dB, *dC;
        cann.allocateDeviceMemory(&dA, M * K * sizeof(float));
        cann.allocateDeviceMemory(&dB, K * N * sizeof(float));
        cann.allocateDeviceMemory(&dC, M * N * sizeof(float));
        
        // Copy to device
        auto stream = cann.createStream();
        cann.memcpyAsync(dA, A, M * K * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
        cann.memcpyAsync(dB, B, K * N * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
        
        // Execute matmul
        // ... (create tensor descriptors and call matmulDirect)
        
        // Copy back
        cann.memcpyAsync(C, dC, M * N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        cann.synchronizeStream(stream);
        
        // Cleanup
        cann.freeDeviceMemory(dA);
        cann.freeDeviceMemory(dB);
        cann.freeDeviceMemory(dC);
        cann.destroyStream(stream);
    }
}

} // namespace npu
} // namespace boas

#endif // BOAS_NPU_DIRECT_ACCESS_H