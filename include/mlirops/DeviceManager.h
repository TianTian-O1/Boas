#ifndef BOAS_DEVICE_MANAGER_H
#define BOAS_DEVICE_MANAGER_H

#include <string>
#include <vector>
#include <memory>

namespace matrix {

/// 设备类型枚举
enum class DeviceType {
    CPU,        // CPU设备
    NPU,        // 华为昇腾NPU
    GPU,        // Nvidia GPU
    UNKNOWN     // 未知设备
};

/// 设备信息结构
struct DeviceInfo {
    DeviceType type;
    int deviceId;
    std::string name;
    std::string properties;
    bool available;
    int priority;  // 优先级（数字越小优先级越高）

    DeviceInfo() : type(DeviceType::UNKNOWN), deviceId(-1),
                   available(false), priority(999) {}
};

/// 设备管理器：负责自动检测和选择最佳计算设备
class DeviceManager {
public:
    /// 获取单例实例
    static DeviceManager& getInstance();

    /// 初始化设备管理器，检测所有可用设备
    bool initialize();

    /// 获取所有可用设备列表
    std::vector<DeviceInfo> getAvailableDevices() const;

    /// 获取当前选中的设备
    DeviceInfo getCurrentDevice() const;

    /// 手动设置设备
    bool setDevice(DeviceType type, int deviceId = 0);

    /// 自动选择最佳设备（按优先级：GPU > NPU > CPU）
    bool selectBestDevice();

    /// 检查特定类型的设备是否可用
    bool isDeviceAvailable(DeviceType type) const;

    /// 获取特定类型设备的数量
    int getDeviceCount(DeviceType type) const;

    /// 获取设备类型的字符串名称
    static std::string deviceTypeToString(DeviceType type);

    /// 从字符串解析设备类型
    static DeviceType stringToDeviceType(const std::string& str);

    /// 打印所有可用设备信息
    void printAvailableDevices() const;

private:
    DeviceManager() = default;
    ~DeviceManager() = default;

    // 禁用拷贝
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    /// 检测NPU设备
    void detectNPUDevices();

    /// 检测GPU设备
    void detectGPUDevices();

    /// 检测CPU设备
    void detectCPUDevices();

    bool initialized_ = false;
    std::vector<DeviceInfo> availableDevices_;
    DeviceInfo currentDevice_;
};

} // namespace matrix

#endif // BOAS_DEVICE_MANAGER_H
