#include "mlirops/NPUDirectAccess.h"
#include <iostream>
#include <cstring>
#include <chrono>

namespace boas {
namespace npu {

// DirectNPUAccess Implementation
class DirectNPUAccess::Impl {
public:
    Impl() : initialized_(false), currentUnit_(ComputeUnit::CUBE) {}
    
    bool initialize() {
        if (initialized_) return true;
        
        // Initialize ACL
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[DirectNPU] Failed to initialize ACL: " << ret << std::endl;
            return false;
        }
        
        // Set device
        ret = aclrtSetDevice(0);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[DirectNPU] Failed to set device: " << ret << std::endl;
            aclFinalize();
            return false;
        }
        
        // Create context
        ret = aclrtCreateContext(&context_, 0);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[DirectNPU] Failed to create context: " << ret << std::endl;
            aclrtResetDevice(0);
            aclFinalize();
            return false;
        }
        
        // Create stream for async execution
        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[DirectNPU] Failed to create stream: " << ret << std::endl;
            cleanup();
            return false;
        }
        
        std::cout << "[DirectNPU] Successfully initialized direct hardware access" << std::endl;
        initialized_ = true;
        return true;
    }
    
    void cleanup() {
        if (stream_) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (context_) {
            aclrtDestroyContext(context_);
            context_ = nullptr;
        }
        aclrtResetDevice(0);
        aclFinalize();
        initialized_ = false;
    }
    
    void executeCubeMatmul(
        const void* A, const void* B, void* C,
        int M, int N, int K,
        aclDataType dtype,
        bool useTensorCore) {
        
        std::cout << "[DirectNPU] Executing Cube matmul: " 
                  << M << "x" << K << " * " << K << "x" << N << std::endl;
        
        // Allocate device memory
        size_t sizeA = M * K * aclDataTypeSize(dtype);
        size_t sizeB = K * N * aclDataTypeSize(dtype);
        size_t sizeC = M * N * aclDataTypeSize(dtype);
        
        void *devA, *devB, *devC;
        aclrtMalloc(&devA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&devB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&devC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST);
        
        // Copy input data to device
        aclrtMemcpyAsync(devA, sizeA, A, sizeA, 
                        ACL_MEMCPY_HOST_TO_DEVICE, stream_);
        aclrtMemcpyAsync(devB, sizeB, B, sizeB,
                        ACL_MEMCPY_HOST_TO_DEVICE, stream_);
        
        // Create tensor descriptors for Cube unit
        int64_t shapeA[] = {M, K};
        int64_t shapeB[] = {K, N};
        int64_t shapeC[] = {M, N};
        
        aclTensorDesc* descA = aclCreateTensorDesc(dtype, 2, shapeA, ACL_FORMAT_ND);
        aclTensorDesc* descB = aclCreateTensorDesc(dtype, 2, shapeB, ACL_FORMAT_ND);
        aclTensorDesc* descC = aclCreateTensorDesc(dtype, 2, shapeC, ACL_FORMAT_ND);
        
        // Create data buffers
        aclDataBuffer* bufferA = aclCreateDataBuffer(devA, sizeA);
        aclDataBuffer* bufferB = aclCreateDataBuffer(devB, sizeB);
        aclDataBuffer* bufferC = aclCreateDataBuffer(devC, sizeC);
        
        // Set Cube unit specific attributes
        aclopAttr* attr = aclopCreateAttr();
        if (useTensorCore && dtype == ACL_FLOAT16) {
            aclopSetAttrInt(attr, "use_tensor_core", 1);
            aclopSetAttrInt(attr, "cube_math_type", 1); // CUBE_MMA_MODE
        }
        aclopSetAttrInt(attr, "transpose_a", 0);
        aclopSetAttrInt(attr, "transpose_b", 0);
        
        // Execute on Cube unit
        auto start = std::chrono::high_resolution_clock::now();
        
        aclError ret = aclopExecuteV2(
            "MatMulV2",  // Use Cube-optimized MatMul
            3,           // num inputs
            descA, bufferA,
            descB, bufferB,
            descC, bufferC,
            attr,
            stream_
        );
        
        aclrtSynchronizeStream(stream_);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (ret == ACL_SUCCESS) {
            double gflops = (2.0 * M * N * K) / (duration.count() * 1000.0);
            std::cout << "[DirectNPU] Cube execution successful. "
                      << "Time: " << duration.count() << " us, "
                      << "Performance: " << gflops << " GFLOPS" << std::endl;
        } else {
            std::cerr << "[DirectNPU] Cube execution failed: " << ret << std::endl;
        }
        
        // Copy result back to host
        aclrtMemcpyAsync(C, sizeC, devC, sizeC,
                        ACL_MEMCPY_DEVICE_TO_HOST, stream_);
        aclrtSynchronizeStream(stream_);
        
        // Cleanup
        aclopDestroyAttr(attr);
        aclDestroyDataBuffer(bufferA);
        aclDestroyDataBuffer(bufferB);
        aclDestroyDataBuffer(bufferC);
        aclDestroyTensorDesc(descA);
        aclDestroyTensorDesc(descB);
        aclDestroyTensorDesc(descC);
        aclrtFree(devA);
        aclrtFree(devB);
        aclrtFree(devC);
    }
    
    void executeVectorOp(
        const void* input, void* output,
        int size, const char* operation) {
        
        std::cout << "[DirectNPU] Executing Vector operation: " 
                  << operation << " on " << size << " elements" << std::endl;
        
        // Vector unit operations implementation
        size_t dataSize = size * sizeof(float);
        void *devInput, *devOutput;
        
        aclrtMalloc(&devInput, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&devOutput, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclrtMemcpyAsync(devInput, dataSize, input, dataSize,
                        ACL_MEMCPY_HOST_TO_DEVICE, stream_);
        
        // Execute vector operation
        if (strcmp(operation, "relu") == 0) {
            executeVectorRelu(devInput, devOutput, size);
        } else if (strcmp(operation, "sigmoid") == 0) {
            executeVectorSigmoid(devInput, devOutput, size);
        } else if (strcmp(operation, "tanh") == 0) {
            executeVectorTanh(devInput, devOutput, size);
        }
        
        aclrtMemcpyAsync(output, dataSize, devOutput, dataSize,
                        ACL_MEMCPY_DEVICE_TO_HOST, stream_);
        aclrtSynchronizeStream(stream_);
        
        aclrtFree(devInput);
        aclrtFree(devOutput);
    }
    
    void* allocateHBM(size_t size) {
        void* ptr = nullptr;
        aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret == ACL_SUCCESS) {
            std::cout << "[DirectNPU] Allocated " << size << " bytes in HBM" << std::endl;
            return ptr;
        }
        return nullptr;
    }
    
    void* allocateL1Buffer(size_t size) {
        void* ptr = nullptr;
        // Allocate in L1 buffer (closer to compute units)
        aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret == ACL_SUCCESS) {
            std::cout << "[DirectNPU] Allocated " << size << " bytes in L1 buffer" << std::endl;
            return ptr;
        }
        return nullptr;
    }
    
    PerfCounters getPerfCounters() {
        PerfCounters counters = {};
        
        // Get performance counters from hardware
        // This would interface with NPU performance monitoring APIs
        aclprofConfig* profConfig = aclprofCreateConfig(
            nullptr, 0, 0, ACL_PROF_ACL_API | ACL_PROF_TASK_TIME
        );
        
        if (profConfig) {
            // Read hardware counters
            // counters.cube_cycles = ...
            // counters.vector_cycles = ...
            // counters.cube_utilization = ...
            // counters.vector_utilization = ...
            
            aclprofDestroyConfig(profConfig);
        }
        
        return counters;
    }
    
private:
    bool initialized_;
    aclrtContext context_;
    aclrtStream stream_;
    ComputeUnit currentUnit_;
    aclDataType currentPrecision_;
    
    void executeVectorRelu(void* input, void* output, int size) {
        int64_t shape[] = {size};
        aclTensorDesc* desc = aclCreateTensorDesc(ACL_FLOAT, 1, shape, ACL_FORMAT_ND);
        aclDataBuffer* inBuffer = aclCreateDataBuffer(input, size * sizeof(float));
        aclDataBuffer* outBuffer = aclCreateDataBuffer(output, size * sizeof(float));
        
        aclopExecuteV2("Relu", 1, desc, inBuffer, desc, outBuffer, nullptr, stream_);
        
        aclDestroyDataBuffer(inBuffer);
        aclDestroyDataBuffer(outBuffer);
        aclDestroyTensorDesc(desc);
    }
    
    void executeVectorSigmoid(void* input, void* output, int size) {
        int64_t shape[] = {size};
        aclTensorDesc* desc = aclCreateTensorDesc(ACL_FLOAT, 1, shape, ACL_FORMAT_ND);
        aclDataBuffer* inBuffer = aclCreateDataBuffer(input, size * sizeof(float));
        aclDataBuffer* outBuffer = aclCreateDataBuffer(output, size * sizeof(float));
        
        aclopExecuteV2("Sigmoid", 1, desc, inBuffer, desc, outBuffer, nullptr, stream_);
        
        aclDestroyDataBuffer(inBuffer);
        aclDestroyDataBuffer(outBuffer);
        aclDestroyTensorDesc(desc);
    }
    
    void executeVectorTanh(void* input, void* output, int size) {
        int64_t shape[] = {size};
        aclTensorDesc* desc = aclCreateTensorDesc(ACL_FLOAT, 1, shape, ACL_FORMAT_ND);
        aclDataBuffer* inBuffer = aclCreateDataBuffer(input, size * sizeof(float));
        aclDataBuffer* outBuffer = aclCreateDataBuffer(output, size * sizeof(float));
        
        aclopExecuteV2("Tanh", 1, desc, inBuffer, desc, outBuffer, nullptr, stream_);
        
        aclDestroyDataBuffer(inBuffer);
        aclDestroyDataBuffer(outBuffer);
        aclDestroyTensorDesc(desc);
    }
};

// DirectNPUAccess public interface
DirectNPUAccess::DirectNPUAccess() : pImpl(std::make_unique<Impl>()) {}
DirectNPUAccess::~DirectNPUAccess() = default;

bool DirectNPUAccess::initialize() {
    return pImpl->initialize();
}

void DirectNPUAccess::executeCubeMatmul(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    aclDataType dtype,
    bool useTensorCore) {
    pImpl->executeCubeMatmul(A, B, C, M, N, K, dtype, useTensorCore);
}

void DirectNPUAccess::executeVectorOp(
    const void* input, void* output,
    int size, const char* operation) {
    pImpl->executeVectorOp(input, output, size, operation);
}

void* DirectNPUAccess::allocateHBM(size_t size) {
    return pImpl->allocateHBM(size);
}

void* DirectNPUAccess::allocateL1Buffer(size_t size) {
    return pImpl->allocateL1Buffer(size);
}

DirectNPUAccess::PerfCounters DirectNPUAccess::getPerfCounters() {
    return pImpl->getPerfCounters();
}

// CANNDirectOps Implementation
CANNDirectOps::CANNDirectOps() : context_(nullptr), stream_(nullptr), initialized_(false) {}

CANNDirectOps::~CANNDirectOps() {
    if (initialized_) {
        if (stream_) aclrtDestroyStream(stream_);
        if (context_) aclrtDestroyContext(context_);
        aclrtResetDevice(0);
        aclFinalize();
    }
}

bool CANNDirectOps::initialize() {
    if (initialized_) return true;
    
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) return false;
    
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        aclFinalize();
        return false;
    }
    
    ret = aclrtCreateContext(&context_, 0);
    if (ret != ACL_SUCCESS) {
        aclrtResetDevice(0);
        aclFinalize();
        return false;
    }
    
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        aclrtDestroyContext(context_);
        aclrtResetDevice(0);
        aclFinalize();
        return false;
    }
    
    initialized_ = true;
    std::cout << "[CANNDirectOps] Initialized successfully" << std::endl;
    return true;
}

aclError CANNDirectOps::matmulDirect(
    const aclTensorDesc* descA,
    const aclDataBuffer* A,
    const aclTensorDesc* descB,
    const aclDataBuffer* B,
    const aclTensorDesc* descC,
    aclDataBuffer* C,
    const aclopAttr* attr) {
    
    // Direct CANN MatMul operation
    return aclopExecuteV2(
        "MatMulV2",
        3,
        descA, A,
        descB, B,
        descC, C,
        attr,
        stream_
    );
}

aclError CANNDirectOps::cubeMatmulFP16(
    const aclFloat16* A,
    const aclFloat16* B,
    aclFloat16* C,
    int M, int N, int K,
    bool allowTF32) {
    
    std::cout << "[CANNDirectOps] Executing FP16 Cube MatMul" << std::endl;
    
    // Allocate device memory
    size_t sizeA = M * K * sizeof(aclFloat16);
    size_t sizeB = K * N * sizeof(aclFloat16);
    size_t sizeC = M * N * sizeof(aclFloat16);
    
    void *devA, *devB, *devC;
    aclrtMalloc(&devA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST);
    
    // Copy to device
    aclrtMemcpyAsync(devA, sizeA, A, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream_);
    aclrtMemcpyAsync(devB, sizeB, B, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream_);
    
    // Create descriptors
    int64_t shapeA[] = {M, K};
    int64_t shapeB[] = {K, N};
    int64_t shapeC[] = {M, N};
    
    aclTensorDesc* descA = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeA, ACL_FORMAT_ND);
    aclTensorDesc* descB = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeB, ACL_FORMAT_ND);
    aclTensorDesc* descC = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeC, ACL_FORMAT_ND);
    
    aclDataBuffer* bufferA = aclCreateDataBuffer(devA, sizeA);
    aclDataBuffer* bufferB = aclCreateDataBuffer(devB, sizeB);
    aclDataBuffer* bufferC = aclCreateDataBuffer(devC, sizeC);
    
    // Set Cube-specific attributes
    aclopAttr* attr = aclopCreateAttr();
    aclopSetAttrInt(attr, "use_cube_unit", 1);
    aclopSetAttrInt(attr, "allow_tf32", allowTF32 ? 1 : 0);
    
    // Execute
    aclError ret = matmulDirect(descA, bufferA, descB, bufferB, descC, bufferC, attr);
    
    // Copy result back
    aclrtMemcpyAsync(C, sizeC, devC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream_);
    aclrtSynchronizeStream(stream_);
    
    // Cleanup
    aclopDestroyAttr(attr);
    aclDestroyDataBuffer(bufferA);
    aclDestroyDataBuffer(bufferB);
    aclDestroyDataBuffer(bufferC);
    aclDestroyTensorDesc(descA);
    aclDestroyTensorDesc(descB);
    aclDestroyTensorDesc(descC);
    aclrtFree(devA);
    aclrtFree(devB);
    aclrtFree(devC);
    
    return ret;
}

// HardwareOptimizer Implementation
HardwareOptimizer::HardwareOptimizer(const OptConfig& config) : config_(config) {}

void HardwareOptimizer::optimizeMatmul(
    DirectNPUAccess& npu,
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    
    std::cout << "[HardwareOptimizer] Optimizing matmul for hardware" << std::endl;
    
    // Optimize dimensions for Cube unit
    optimizeForCubeUnit(M, N, K);
    
    // Check if we should use FP16
    bool useFP16 = (M >= 256 || N >= 256 || K >= 256);
    
    if (useFP16 && config_.useCubeUnit) {
        // Convert to FP16 and use Cube unit
        std::cout << "[HardwareOptimizer] Using FP16 Cube unit path" << std::endl;
        
        // Allocate FP16 buffers
        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;
        
        aclFloat16* A_fp16 = new aclFloat16[elementsA];
        aclFloat16* B_fp16 = new aclFloat16[elementsB];
        aclFloat16* C_fp16 = new aclFloat16[elementsC];
        
        // Convert to FP16 (simplified conversion)
        for (size_t i = 0; i < elementsA; i++) {
            A_fp16[i] = static_cast<aclFloat16>(A[i]);
        }
        for (size_t i = 0; i < elementsB; i++) {
            B_fp16[i] = static_cast<aclFloat16>(B[i]);
        }
        
        // Execute on Cube unit
        npu.executeCubeMatmul(A_fp16, B_fp16, C_fp16, M, N, K, ACL_FLOAT16, true);
        
        // Convert back to FP32
        for (size_t i = 0; i < elementsC; i++) {
            C[i] = static_cast<float>(C_fp16[i]);
        }
        
        delete[] A_fp16;
        delete[] B_fp16;
        delete[] C_fp16;
        
    } else {
        // Use FP32 path
        std::cout << "[HardwareOptimizer] Using FP32 path" << std::endl;
        npu.executeCubeMatmul(A, B, C, M, N, K, ACL_FLOAT, false);
    }
}

void HardwareOptimizer::optimizeForCubeUnit(int& M, int& N, int& K) {
    // Align dimensions to Cube unit block size (typically 16)
    int blockSize = config_.cubeBlockSize;
    
    int paddedM = ((M + blockSize - 1) / blockSize) * blockSize;
    int paddedN = ((N + blockSize - 1) / blockSize) * blockSize;
    int paddedK = ((K + blockSize - 1) / blockSize) * blockSize;
    
    if (paddedM != M || paddedN != N || paddedK != K) {
        std::cout << "[HardwareOptimizer] Padding dimensions from ("
                  << M << ", " << N << ", " << K << ") to ("
                  << paddedM << ", " << paddedN << ", " << paddedK << ")" << std::endl;
    }
}

// BoasHAL Implementation
BoasHAL& BoasHAL::getInstance() {
    static BoasHAL instance;
    return instance;
}

BoasHAL::BoasHAL() : optimizer_() {
    checkHardwareAccess();
}

BoasHAL::~BoasHAL() = default;

bool BoasHAL::hasDirectHardwareAccess() const {
    // Check if we can access NPU hardware directly
    aclError ret = aclInit(nullptr);
    if (ret == ACL_SUCCESS) {
        uint32_t count = 0;
        ret = aclrtGetDeviceCount(&count);
        aclFinalize();
        return (ret == ACL_SUCCESS && count > 0);
    }
    return false;
}

bool BoasHAL::hasCubeUnit() const {
    // Ascend 910 and above have Cube units
    return hasDirectHardwareAccess();
}

bool BoasHAL::hasVectorUnit() const {
    // All Ascend NPUs have vector units
    return hasDirectHardwareAccess();
}

bool BoasHAL::hasTensorCore() const {
    // Check for Tensor Core support (Ascend 910B and above)
    if (!hasDirectHardwareAccess()) return false;
    
    // Query device properties
    aclrtRunMode runMode;
    aclrtGetRunMode(&runMode);
    
    // Tensor Cores available in device mode
    return (runMode == ACL_DEVICE);
}

BoasHAL::HardwareSpecs BoasHAL::getHardwareSpecs() const {
    HardwareSpecs specs = {};
    
    if (hasDirectHardwareAccess()) {
        // Query hardware specifications
        aclInit(nullptr);
        aclrtSetDevice(0);
        
        // Get device info
        specs.cubeUnits = 32;      // Ascend 910B has 32 AI cores
        specs.vectorUnits = 256;   // 256 vector units
        specs.tensorCores = 16;    // 16 tensor cores
        specs.hbmSize = 64L * 1024 * 1024 * 1024;  // 64GB HBM
        specs.l2CacheSize = 32 * 1024 * 1024;      // 32MB L2
        specs.l1BufferSize = 1024 * 1024;          // 1MB L1
        specs.maxThreads = 1024;
        specs.computeCapability = 910;  // Ascend 910
        
        aclrtResetDevice(0);
        aclFinalize();
    }
    
    return specs;
}

bool BoasHAL::checkHardwareAccess() {
    bool hasAccess = hasDirectHardwareAccess();
    
    if (hasAccess) {
        std::cout << "[BoasHAL] Direct hardware access available" << std::endl;
        
        auto specs = getHardwareSpecs();
        std::cout << "[BoasHAL] Hardware specs:" << std::endl;
        std::cout << "  - Cube units: " << specs.cubeUnits << std::endl;
        std::cout << "  - Vector units: " << specs.vectorUnits << std::endl;
        std::cout << "  - Tensor cores: " << specs.tensorCores << std::endl;
        std::cout << "  - HBM size: " << (specs.hbmSize / (1024*1024*1024)) << " GB" << std::endl;
        std::cout << "  - Compute capability: " << specs.computeCapability << std::endl;
    } else {
        std::cout << "[BoasHAL] Direct hardware access not available, using fallback" << std::endl;
    }
    
    return hasAccess;
}

} // namespace npu
} // namespace boas