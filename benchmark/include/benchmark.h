#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_init.h>
#include <mach/task.h>
#include <mach/task_info.h>
#endif

class Benchmark {
public:
    struct Result {
        std::string name;
        std::string language;
        size_t size;
        double time_ms;
        double memory_kb;
    };

    static void run(const std::string& name, 
                   const std::string& language,
                   size_t size,
                   std::function<void()> func) {
        // 记录开始内存
        double start_memory = getCurrentMemoryUsage();
        
        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();
        
        // 运行测试
        func();
        
        // 记录结束时间和内存
        auto end = std::chrono::high_resolution_clock::now();
        double end_memory = getCurrentMemoryUsage();
        
        // 计算结果
        double time_ms = std::chrono::duration_cast<std::chrono::microseconds>
            (end - start).count() / 1000.0;
        double memory_kb = (end_memory - start_memory);
        
        // 打印结果
        std::cout << "C++ benchmark (size=" << size << "): "
                  << "Time=" << time_ms << "ms, "
                  << "Memory=" << memory_kb << "KB" << std::endl;
        
        // 保存结果
        Result result{name, language, size, time_ms, memory_kb};
        saveResult(result);
    }

private:
    static double getCurrentMemoryUsage() {
        #ifdef __APPLE__
            struct task_basic_info info;
            mach_msg_type_number_t size = sizeof(info);
            kern_return_t kerr = task_info(mach_task_self(),
                                         TASK_BASIC_INFO,
                                         (task_info_t)&info,
                                         &size);
            if (kerr == KERN_SUCCESS) {
                return static_cast<double>(info.resident_size) / 1024.0; // 转换为 KB
            }
        #endif
        return 0.0;
    }

    static void saveResult(const Result& result) {
        std::ofstream file("../results/results.csv", std::ios::app);
        if (file.is_open()) {
            file << result.name << ","
                 << result.language << ","
                 << result.size << ","
                 << std::fixed << std::setprecision(2) << result.time_ms << ","
                 << std::fixed << std::setprecision(2) << result.memory_kb << "\n";
        }
    }
};
