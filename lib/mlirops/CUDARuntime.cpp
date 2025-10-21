#include "mlirops/CUDARuntime.h"
#include <sstream>
#include <cstring>

namespace matrix {

CUDARuntime& CUDARuntime::getInstance() {
    static CUDARuntime instance;
    return instance;
}

bool CUDARuntime::initialize() {
    if (initialized_) {
        return true;
    }

    std::cout << "[CUDA] Initializing CUDA runtime..." << std::endl;

#ifdef HAVE_CUDA_SUPPORT
    // 获取设备数量
    cudaError_t ret = cudaGetDeviceCount(&deviceCount_);
    if (!checkCudaError(ret, "cudaGetDeviceCount")) {
        return false;
    }

    std::cout << "[CUDA] Found " << deviceCount_ << " GPU device(s)" << std::endl;

    if (deviceCount_ == 0) {
        lastError_ = "No CUDA-capable GPU devices found";
        std::cerr << "[CUDA] Error: " << lastError_ << std::endl;
        return false;
    }

    // 设置默认设备
    if (!setDevice(0)) {
        return false;
    }

    // 初始化cuBLAS
    cublasStatus_t blasStatus = cublasCreate(&cublasHandle_);
    if (!checkCublasError(blasStatus, "cublasCreate")) {
        return false;
    }

    std::cout << "[CUDA] Successfully initialized with device 0" << std::endl;
    std::cout << "[CUDA] Device properties: " << getDeviceProperties(0) << std::endl;

    initialized_ = true;
    return true;

#else
    lastError_ = "CUDA support not compiled";
    std::cerr << "[CUDA] Error: " << lastError_ << std::endl;
    std::cerr << "[CUDA] Please compile with -DHAVE_CUDA_SUPPORT and link CUDA libraries" << std::endl;
    return false;
#endif
}

void CUDARuntime::finalize() {
    if (!initialized_) {
        return;
    }

    std::cout << "[CUDA] Finalizing CUDA runtime..." << std::endl;

#ifdef HAVE_CUDA_SUPPORT
    // 销毁cuBLAS句柄
    if (cublasHandle_ != nullptr) {
        cublasDestroy(cublasHandle_);
        cublasHandle_ = nullptr;
    }

    // 重置设备
    cudaDeviceReset();
#endif

    initialized_ = false;
    std::cout << "[CUDA] CUDA runtime finalized" << std::endl;
}

CUDARuntime::~CUDARuntime() {
    finalize();
}

bool CUDARuntime::isAvailable() const {
#ifdef HAVE_CUDA_SUPPORT
    return initialized_;
#else
    return false;
#endif
}

int CUDARuntime::getDeviceCount() const {
    return deviceCount_;
}

bool CUDARuntime::setDevice(int deviceId) {
#ifdef HAVE_CUDA_SUPPORT
    if (deviceId < 0 || deviceId >= deviceCount_) {
        lastError_ = "Invalid device ID: " + std::to_string(deviceId);
        std::cerr << "[CUDA] Error: " << lastError_ << std::endl;
        return false;
    }

    cudaError_t ret = cudaSetDevice(deviceId);
    if (!checkCudaError(ret, "cudaSetDevice")) {
        return false;
    }

    currentDevice_ = deviceId;
    std::cout << "[CUDA] Set current device to " << deviceId << std::endl;
    return true;
#else
    lastError_ = "CUDA support not compiled";
    return false;
#endif
}

int CUDARuntime::getCurrentDevice() const {
    return currentDevice_;
}

std::string CUDARuntime::getDeviceProperties(int deviceId) const {
#ifdef HAVE_CUDA_SUPPORT
    cudaDeviceProp prop;
    cudaError_t ret = cudaGetDeviceProperties(&prop, deviceId);
    if (ret != cudaSuccess) {
        return "Unknown GPU";
    }

    std::stringstream ss;
    ss << prop.name
       << " (Compute " << prop.major << "." << prop.minor
       << ", " << (prop.totalGlobalMem / (1024*1024)) << " MB"
       << ", " << prop.multiProcessorCount << " SMs)";
    return ss.str();
#else
    return "CUDA not available";
#endif
}

void* CUDARuntime::allocateMemory(size_t size) {
#ifdef HAVE_CUDA_SUPPORT
    void* ptr = nullptr;
    cudaError_t ret = cudaMalloc(&ptr, size);
    if (!checkCudaError(ret, "cudaMalloc")) {
        return nullptr;
    }
    return ptr;
#else
    lastError_ = "CUDA support not compiled";
    return nullptr;
#endif
}

bool CUDARuntime::freeMemory(void* ptr) {
#ifdef HAVE_CUDA_SUPPORT
    if (ptr == nullptr) {
        return true;
    }
    cudaError_t ret = cudaFree(ptr);
    return checkCudaError(ret, "cudaFree");
#else
    lastError_ = "CUDA support not compiled";
    return false;
#endif
}

bool CUDARuntime::copyHostToDevice(void* dst, const void* src, size_t size) {
#ifdef HAVE_CUDA_SUPPORT
    cudaError_t ret = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return checkCudaError(ret, "cudaMemcpy H2D");
#else
    lastError_ = "CUDA support not compiled";
    return false;
#endif
}

bool CUDARuntime::copyDeviceToHost(void* dst, const void* src, size_t size) {
#ifdef HAVE_CUDA_SUPPORT
    cudaError_t ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return checkCudaError(ret, "cudaMemcpy D2H");
#else
    lastError_ = "CUDA support not compiled";
    return false;
#endif
}

bool CUDARuntime::executeMatmul(
    const float* a, const float* b, float* c,
    int m, int k, int n,
    bool transposeA, bool transposeB) {

#ifdef HAVE_CUDA_SUPPORT
    if (!initialized_) {
        lastError_ = "CUDA runtime not initialized";
        return false;
    }

    // cuBLAS使用列主序，所以需要调整
    // C = A * B (行主序) 等价于 C^T = B^T * A^T (列主序)
    // 所以在cuBLAS中我们计算 C = B * A

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasOperation_t opA = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;

    // cublasSgemm: C = alpha * op(B) * op(A) + beta * C
    // 其中 op(B) 是 n x k, op(A) 是 k x m, C 是 n x m
    cublasStatus_t status = cublasSgemm(
        cublasHandle_,
        opA,           // op(B)
        opB,           // op(A)
        n,             // C的行数
        m,             // C的列数
        k,             // 内部维度
        &alpha,
        b, transposeB ? k : n,  // B的leading dimension
        a, transposeA ? m : k,  // A的leading dimension
        &beta,
        c, n           // C的leading dimension
    );

    return checkCublasError(status, "cublasSgemm");
#else
    lastError_ = "CUDA support not compiled";
    return false;
#endif
}

bool CUDARuntime::executeCustomMatmul(
    const float* a, const float* b, float* c,
    int m, int k, int n) {

#ifdef HAVE_CUDA_SUPPORT
    // 这里将调用自定义CUDA kernel
    // 在后续步骤中实现
    lastError_ = "Custom CUDA kernel not yet implemented";
    std::cerr << "[CUDA] " << lastError_ << std::endl;
    std::cerr << "[CUDA] Falling back to cuBLAS implementation" << std::endl;
    return executeMatmul(a, b, c, m, k, n);
#else
    lastError_ = "CUDA support not compiled";
    return false;
#endif
}

bool CUDARuntime::synchronize() {
#ifdef HAVE_CUDA_SUPPORT
    cudaError_t ret = cudaDeviceSynchronize();
    return checkCudaError(ret, "cudaDeviceSynchronize");
#else
    lastError_ = "CUDA support not compiled";
    return false;
#endif
}

std::string CUDARuntime::getLastError() const {
    return lastError_;
}

#ifdef HAVE_CUDA_SUPPORT
std::string CUDARuntime::cudaErrorToString(cudaError_t error) const {
    return cudaGetErrorString(error);
}

std::string CUDARuntime::cublasErrorToString(cublasStatus_t status) const {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "UNKNOWN_CUBLAS_ERROR";
    }
}

bool CUDARuntime::checkCudaError(cudaError_t error, const std::string& operation) const {
    if (error != cudaSuccess) {
        lastError_ = operation + " failed: " + cudaErrorToString(error);
        std::cerr << "[CUDA] Error: " << lastError_ << std::endl;
        return false;
    }
    return true;
}

bool CUDARuntime::checkCublasError(cublasStatus_t status, const std::string& operation) const {
    if (status != CUBLAS_STATUS_SUCCESS) {
        lastError_ = operation + " failed: " + cublasErrorToString(status);
        std::cerr << "[CUDA] Error: " << lastError_ << std::endl;
        return false;
    }
    return true;
}
#endif

// GPUMemoryBuffer implementation
GPUMemoryBuffer::GPUMemoryBuffer(size_t size) : size_(size) {
    auto& runtime = CUDARuntime::getInstance();
    ptr_ = runtime.allocateMemory(size);
    if (ptr_ == nullptr) {
        throw std::runtime_error("Failed to allocate GPU memory: " + runtime.getLastError());
    }
}

GPUMemoryBuffer::~GPUMemoryBuffer() {
    if (ptr_ != nullptr) {
        auto& runtime = CUDARuntime::getInstance();
        runtime.freeMemory(ptr_);
        ptr_ = nullptr;
    }
}

bool GPUMemoryBuffer::copyFromHost(const void* src, size_t copySize) {
    auto& runtime = CUDARuntime::getInstance();
    size_t actualSize = (copySize == 0) ? size_ : std::min(copySize, size_);
    return runtime.copyHostToDevice(ptr_, src, actualSize);
}

bool GPUMemoryBuffer::copyToHost(void* dst, size_t copySize) const {
    auto& runtime = CUDARuntime::getInstance();
    size_t actualSize = (copySize == 0) ? size_ : std::min(copySize, size_);
    return runtime.copyDeviceToHost(dst, ptr_, actualSize);
}

} // namespace matrix
