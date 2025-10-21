#include "mlirops/DeviceManager.h"
#include "mlirops/NPUBackend.h"
#include "mlirops/GPUBackend.h"
#include <iostream>
#include <algorithm>

namespace matrix {

DeviceManager& DeviceManager::getInstance() {
    static DeviceManager instance;
    return instance;
}

bool DeviceManager::initialize() {
    if (initialized_) {
        return true;
    }

    std::cout << "[DeviceManager] Initializing device manager..." << std::endl;
    std::cout << "[DeviceManager] Scanning for available devices..." << std::endl;

    availableDevices_.clear();

    // 检测各类设备
    detectGPUDevices();
    detectNPUDevices();
    detectCPUDevices();

    // 按优先级排序（数字越小优先级越高）
    std::sort(availableDevices_.begin(), availableDevices_.end(),
              [](const DeviceInfo& a, const DeviceInfo& b) {
                  return a.priority < b.priority;
              });

    std::cout << "[DeviceManager] Found " << availableDevices_.size()
              << " available device(s)" << std::endl;

    initialized_ = true;

    // 自动选择最佳设备
    selectBestDevice();

    return true;
}

void DeviceManager::detectGPUDevices() {
    std::cout << "[DeviceManager] Detecting Nvidia GPU devices..." << std::endl;

    if (GPUBackend::isAvailable()) {
        int count = GPUBackend::getDeviceCount();
        std::cout << "[DeviceManager] Found " << count << " GPU device(s)" << std::endl;

        for (int i = 0; i < count; i++) {
            DeviceInfo info;
            info.type = DeviceType::GPU;
            info.deviceId = i;
            info.name = "Nvidia GPU " + std::to_string(i);

            // 临时设置设备以获取属性
            GPUBackend::setDevice(i);
            info.properties = GPUBackend::getDeviceProperties();

            info.available = true;
            info.priority = 10 + i;  // GPU优先级最高（10-19）

            availableDevices_.push_back(info);
            std::cout << "[DeviceManager]   GPU " << i << ": " << info.properties << std::endl;
        }
    } else {
        std::cout << "[DeviceManager] No GPU devices found or CUDA not available" << std::endl;
    }
}

void DeviceManager::detectNPUDevices() {
    std::cout << "[DeviceManager] Detecting Ascend NPU devices..." << std::endl;

    if (NPUBackend::isAvailable()) {
        int count = NPUBackend::getDeviceCount();
        std::cout << "[DeviceManager] Found " << count << " NPU device(s)" << std::endl;

        for (int i = 0; i < count; i++) {
            DeviceInfo info;
            info.type = DeviceType::NPU;
            info.deviceId = i;
            info.name = "Ascend NPU " + std::to_string(i);

            // 临时设置设备以获取属性
            NPUBackend::setDevice(i);
            info.properties = NPUBackend::getDeviceProperties();

            info.available = true;
            info.priority = 20 + i;  // NPU优先级次之（20-29）

            availableDevices_.push_back(info);
            std::cout << "[DeviceManager]   NPU " << i << ": " << info.properties << std::endl;
        }
    } else {
        std::cout << "[DeviceManager] No NPU devices found or CANN not available" << std::endl;
    }
}

void DeviceManager::detectCPUDevices() {
    std::cout << "[DeviceManager] Detecting CPU..." << std::endl;

    // CPU总是可用
    DeviceInfo info;
    info.type = DeviceType::CPU;
    info.deviceId = 0;
    info.name = "CPU";
    info.properties = "Host CPU (fallback)";
    info.available = true;
    info.priority = 100;  // CPU优先级最低

    availableDevices_.push_back(info);
    std::cout << "[DeviceManager]   CPU: Available (fallback mode)" << std::endl;
}

std::vector<DeviceInfo> DeviceManager::getAvailableDevices() const {
    return availableDevices_;
}

DeviceInfo DeviceManager::getCurrentDevice() const {
    return currentDevice_;
}

bool DeviceManager::setDevice(DeviceType type, int deviceId) {
    // 查找指定的设备
    auto it = std::find_if(availableDevices_.begin(), availableDevices_.end(),
                          [type, deviceId](const DeviceInfo& info) {
                              return info.type == type && info.deviceId == deviceId;
                          });

    if (it == availableDevices_.end()) {
        std::cerr << "[DeviceManager] Device not found: "
                  << deviceTypeToString(type) << " " << deviceId << std::endl;
        return false;
    }

    if (!it->available) {
        std::cerr << "[DeviceManager] Device not available: "
                  << deviceTypeToString(type) << " " << deviceId << std::endl;
        return false;
    }

    // 设置后端设备
    bool success = false;
    switch (type) {
        case DeviceType::GPU:
            success = GPUBackend::setDevice(deviceId);
            break;
        case DeviceType::NPU:
            success = NPUBackend::setDevice(deviceId);
            break;
        case DeviceType::CPU:
            success = true;  // CPU总是可用
            break;
        default:
            std::cerr << "[DeviceManager] Unknown device type" << std::endl;
            return false;
    }

    if (success) {
        currentDevice_ = *it;
        std::cout << "[DeviceManager] Current device set to: "
                  << currentDevice_.name << " (" << currentDevice_.properties << ")" << std::endl;
    }

    return success;
}

bool DeviceManager::selectBestDevice() {
    if (availableDevices_.empty()) {
        std::cerr << "[DeviceManager] No devices available!" << std::endl;
        return false;
    }

    // 设备已按优先级排序，选择第一个
    const auto& bestDevice = availableDevices_[0];

    std::cout << "[DeviceManager] Auto-selecting best device: "
              << bestDevice.name << " (" << deviceTypeToString(bestDevice.type) << ")" << std::endl;

    return setDevice(bestDevice.type, bestDevice.deviceId);
}

bool DeviceManager::isDeviceAvailable(DeviceType type) const {
    return std::any_of(availableDevices_.begin(), availableDevices_.end(),
                      [type](const DeviceInfo& info) {
                          return info.type == type && info.available;
                      });
}

int DeviceManager::getDeviceCount(DeviceType type) const {
    return std::count_if(availableDevices_.begin(), availableDevices_.end(),
                        [type](const DeviceInfo& info) {
                            return info.type == type && info.available;
                        });
}

std::string DeviceManager::deviceTypeToString(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::NPU: return "NPU";
        case DeviceType::GPU: return "GPU";
        default: return "Unknown";
    }
}

DeviceType DeviceManager::stringToDeviceType(const std::string& str) {
    if (str == "CPU" || str == "cpu") return DeviceType::CPU;
    if (str == "NPU" || str == "npu") return DeviceType::NPU;
    if (str == "GPU" || str == "gpu") return DeviceType::GPU;
    return DeviceType::UNKNOWN;
}

void DeviceManager::printAvailableDevices() const {
    std::cout << "\n=== Available Devices ===" << std::endl;
    std::cout << "Total: " << availableDevices_.size() << " device(s)\n" << std::endl;

    for (size_t i = 0; i < availableDevices_.size(); i++) {
        const auto& dev = availableDevices_[i];
        std::cout << "[" << i << "] " << deviceTypeToString(dev.type)
                  << " " << dev.deviceId << ": " << dev.properties;

        if (dev.deviceId == currentDevice_.deviceId &&
            dev.type == currentDevice_.type) {
            std::cout << " [CURRENT]";
        }
        std::cout << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

} // namespace matrix
