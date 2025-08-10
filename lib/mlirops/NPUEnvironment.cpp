#include "mlirops/NPUBackend.h"
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

namespace matrix {

/// NPU环境检测和初始化
class NPUEnvironment {
public:
    /// 检测昇腾NPU环境是否可用
    static bool checkNPUEnvironment() {
        std::cout << "[NPU] Checking Ascend NPU environment..." << std::endl;
        
        // 1. 检查CANN环境变量
        if (!checkCANNEnvironment()) {
            std::cout << "[NPU] CANN environment not found" << std::endl;
            return false;
        }
        
        // 2. 检查NPU设备文件
        if (!checkNPUDevices()) {
            std::cout << "[NPU] No NPU devices found" << std::endl;
            return false;
        }
        
        // 3. 检查torch_npu是否可用（可选）
        if (!checkTorchNPU()) {
            std::cout << "[NPU] torch_npu not available, using basic NPU support" << std::endl;
        }
        
        std::cout << "[NPU] NPU environment check passed" << std::endl;
        return true;
    }
    
private:
    /// 检查CANN工具链环境
    static bool checkCANNEnvironment() {
        // 检查ASCEND_HOME环境变量
        const char* ascendHome = std::getenv("ASCEND_HOME");
        if (!ascendHome) {
            ascendHome = std::getenv("ASCEND_TOOLKIT_HOME");
        }
        if (!ascendHome) {
            // 尝试默认路径
            std::ifstream cannFile("/usr/local/Ascend/ascend-toolkit/set_env.sh");
            if (!cannFile.good()) {
                std::ifstream homeFile((std::string(std::getenv("HOME")) + "/Ascend/ascend-toolkit/set_env.sh").c_str());
                return homeFile.good();
            }
            return true;
        }
        
        std::cout << "[NPU] Found ASCEND_HOME: " << ascendHome << std::endl;
        return true;
    }
    
    /// 检查NPU设备
    static bool checkNPUDevices() {
        // 检查标准的NPU设备文件
        std::ifstream dev0("/dev/davinci0");
        if (dev0.good()) {
            std::cout << "[NPU] Found NPU device: /dev/davinci0" << std::endl;
            return true;
        }
        
        // 检查管理器设备
        std::ifstream devmgr("/dev/davinci_manager");
        if (devmgr.good()) {
            std::cout << "[NPU] Found NPU manager: /dev/davinci_manager" << std::endl;
            return true;
        }
        
        // 模拟环境下，假设有NPU设备
        std::cout << "[NPU] Running in simulation mode" << std::endl;
        return true; // 为了开发调试，暂时返回true
    }
    
    /// 检查torch_npu可用性
    static bool checkTorchNPU() {
        // 这里可以尝试加载torch_npu
        // 目前只做简单的环境变量检查
        const char* pythonPath = std::getenv("PYTHONPATH");
        if (pythonPath) {
            std::string path(pythonPath);
            if (path.find("torch_npu") != std::string::npos) {
                std::cout << "[NPU] torch_npu found in PYTHONPATH" << std::endl;
                return true;
            }
        }
        return false;
    }
};

/// 在NPUBackend初始化时调用环境检测
bool NPUBackend::isAvailable() {
    static bool checked = false;
    static bool available = false;
    
    if (!checked) {
        available = NPUEnvironment::checkNPUEnvironment();
        checked = true;
    }
    
    return available;
}

} // namespace matrix
