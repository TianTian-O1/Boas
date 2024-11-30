#include <iostream>
#include <string>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <cstdio>
#include <memory>
#endif

class PerformanceAnalyzer {
public:
    static double getMemoryUsage() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
        return static_cast<double>(pmc.WorkingSetSize) / 1024.0; // 转换为KB
#else
        FILE* file = fopen("/proc/self/status", "r");
        if (file) {
            char line[128];
            while (fgets(line, 128, file) != NULL) {
                if (strncmp(line, "VmRSS:", 6) == 0) {
                    long rss;
                    sscanf(line, "VmRSS: %ld", &rss);
                    fclose(file);
                    return static_cast<double>(rss);
                }
            }
            fclose(file);
        }
        return 0.0;
#endif
    }

    static double getCPUUsage() {
#ifdef _WIN32
        static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
        static int numProcessors;
        static bool init = false;

        if (!init) {
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            numProcessors = sysInfo.dwNumberOfProcessors;

            GetSystemTimeAsFileTime((FILETIME*)&lastCPU);
            GetProcessTimes(GetCurrentProcess(), (FILETIME*)&lastCPU,
                          (FILETIME*)&lastCPU, (FILETIME*)&lastSysCPU,
                          (FILETIME*)&lastUserCPU);
            init = true;
            return 0.0;
        }

        FILETIME createTime, exitTime, kernelTime, userTime;
        ULARGE_INTEGER now, sys, user;

        GetSystemTimeAsFileTime((FILETIME*)&now);
        GetProcessTimes(GetCurrentProcess(), &createTime, &exitTime,
                       &kernelTime, &userTime);

        sys.LowPart = kernelTime.dwLowDateTime;
        sys.HighPart = kernelTime.dwHighDateTime;
        user.LowPart = userTime.dwLowDateTime;
        user.HighPart = userTime.dwHighDateTime;

        double percent = (sys.QuadPart - lastSysCPU.QuadPart) +
                        (user.QuadPart - lastUserCPU.QuadPart);
        percent /= (now.QuadPart - lastCPU.QuadPart);
        percent /= numProcessors;
        percent *= 100;

        lastCPU = now;
        lastUserCPU = user;
        lastSysCPU = sys;

        return percent;
#else
        return 0.0; // 在非Windows系统上返回0
#endif
    }
};

int main() {
    // 每秒记录一次性能数据
    for (int i = 0; i < 10; ++i) {
        double memory = PerformanceAnalyzer::getMemoryUsage();
        double cpu = PerformanceAnalyzer::getCPUUsage();
        
        std::cout << "Memory Usage: " << memory << " KB, CPU Usage: " 
                  << cpu << "%" << std::endl;
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
} 