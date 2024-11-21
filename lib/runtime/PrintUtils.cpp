#include <stdio.h>
#include <stdint.h>
#include <chrono>

extern "C" {

// 打印浮点数
void printFloat(double value) {
    // 如果是时间戳（大于1e6），转换为日期时间格式
    if (value > 1e6) {
        // 将double转换为毫秒
        auto milliseconds = static_cast<int64_t>(value);
        
        // 转换为时间点
        auto timePoint = std::chrono::system_clock::time_point(
            std::chrono::milliseconds(milliseconds)
        );
        
        // 转换为时间
        auto time = std::chrono::system_clock::to_time_t(timePoint);
        auto localTime = *std::localtime(&time);
        
        // 获取毫秒部分
        auto ms = milliseconds % 1000;
        
        // 格式化输出
        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &localTime);
        printf("%s.%03ld\n", buffer, ms);
    } else {
        printf("%g\n", value);    // 其他数字使用通用格式
    }
}

// 打印字符串
void printString(char* str) {
    printf("%s\n", str);
}

// 打印整数
void printInt(int64_t value) {
    printf("%ld\n", value);
}

double system_time_msec() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

} 