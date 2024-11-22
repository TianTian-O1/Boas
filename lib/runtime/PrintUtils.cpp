#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <random>
#include <thread>
#include <string>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace {
    // 获取当前时间戳的格式化字符串
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()
        ).count() % 1000;
        
        auto time = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time);
        
        std::ostringstream oss;
        oss << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") 
            << "." << std::setfill('0') << std::setw(3) << ms << "] ";
            
        return oss.str();
    }

    // 打印矩阵相关的全局状态
    struct PrintState {
        int count = 0;          // 当前打印的元素计数
        int totalElements = 0;  // 矩阵总元素数
        int matrixSize = 0;     // 矩阵维度（假设是方阵）
        bool headerPrinted = false; // 是否已打印矩阵头部
    };

    static PrintState printState;

    void resetPrintState() {
        printState = PrintState();
    }
}

extern "C" {

void printFloat(double value) {
    // 第一个数字表示总元素个数
    if (printState.count == 0) {
        printState.totalElements = static_cast<int>(value);
        printState.matrixSize = static_cast<int>(sqrt(printState.totalElements));
        printf("%sMatrix %dx%d:\n", getCurrentTimestamp().c_str(), 
               printState.matrixSize, printState.matrixSize);
        printf("[\n");
        printState.headerPrinted = true;
    } else {
        // 打印矩阵元素
        if ((printState.count - 1) % printState.matrixSize == 0) {
            printf("  "); // 每行开始的缩进
        }
        
        // 打印数字，使用固定宽度格式
        printf("%8.4f", value);
        
        // 添加分隔符
        if ((printState.count - 1) % printState.matrixSize == printState.matrixSize - 1) {
            if (printState.count < printState.totalElements) {
                printf(",\n"); // 行末，但不是最后一行
            } else {
                printf("\n"); // 最后一行
            }
        } else if (printState.count < printState.totalElements) {
            printf(", "); // 同一行的元素之间
        }
        
        // 检查是否是矩阵的最后一个元素
        if (printState.count == printState.totalElements) {
            printf("]\n"); // 关闭矩阵的方括号
            resetPrintState(); // 重置状态，为下一个矩阵做准备
        }
    }
    
    printState.count++;
    fflush(stdout);
}

void printString(const char* str) {
    printf("%s%s\n", getCurrentTimestamp().c_str(), str);
    fflush(stdout);
}

void printInt(int64_t value) {
    printf("%s%lld\n", getCurrentTimestamp().c_str(), value);
    fflush(stdout);
}

void printError(const char* message) {
    printf("%sError: %s\n", getCurrentTimestamp().c_str(), message);
    fflush(stdout);
}

void printDebug(const char* message) {
    printf("%sDebug: %s\n", getCurrentTimestamp().c_str(), message);
    fflush(stdout);
}

void printSeparator() {
    printf("\n%s----------------------------------------\n", getCurrentTimestamp().c_str());
    fflush(stdout);
}

double system_time_msec() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

double generate_random() {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

} // extern "C"