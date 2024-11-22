#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <random>
#include <thread>
#include <string>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <vector>
#include <deque>

namespace {
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

    // Helper function to check if a value is exactly an integer
    bool isExactInteger(double value) {
        return std::abs(value - std::round(value)) < 1e-10;
    }

    // Helper function to check if a value is a benchmark marker
    bool isBenchmarkMarker(double value) {
        // 只有在基准测试上下文中的数字才被视为标记
        // 这里我们可以通过一个静态标志来追踪上下文
        static bool inBenchmarkContext = false;
        
        if (isExactInteger(value)) {
            if (value >= 1.0 && value <= 10.0 && value == std::floor(value)) {
                inBenchmarkContext = true;
                return true;
            } else if (inBenchmarkContext) {
                // 如果已经在基准测试上下文中，重置标志
                inBenchmarkContext = false;
            }
        }
        return false;
    }

    // Simple printer for regular numbers
    void printNumber(double value) {
        printf("%s%8.4f\n", getCurrentTimestamp().c_str(), value);
        fflush(stdout);
    }

    class BenchmarkTracker {
    public:
        void handleValue(double value) {
            if (isBenchmarkMarker(value)) {
                // This is a benchmark marker
                int marker = static_cast<int>(value);
                
                if (marker % 2 == 1) {
                    // Odd numbers are start markers
                    printf("%s=== Starting benchmark case %d ===\n", 
                           getCurrentTimestamp().c_str(), (marker + 1) / 2);
                } else {
                    // Even numbers are end markers
                    printf("%s=== Completed benchmark case %d ===\n", 
                           getCurrentTimestamp().c_str(), marker / 2);
                    
                    // Print time taken if we have a start time
                    if (startTime > 0) {
                        double duration = getCurrentTimeMs() - startTime;
                        printf("%sTime taken: %.2f ms\n", 
                               getCurrentTimestamp().c_str(), duration);
                        startTime = 0;  // Reset for next benchmark
                    }
                }
                
                if (marker % 2 == 1) {
                    // Start timing for odd numbers (start markers)
                    startTime = getCurrentTimeMs();
                }
            } else {
                // Regular number, just print it
                printNumber(value);
            }
            fflush(stdout);
        }

    private:
        double startTime = 0;

        double getCurrentTimeMs() {
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        }
    };

    static BenchmarkTracker benchmarkTracker;

    // 新增：矩阵打印状态追踪
    class MatrixPrintState {
    private:
        int currentRow = 0;
        int currentCol = 0;
        int matrixSize = 0;
        bool isFirstElement = true;
        
    public:
        void reset(int size = 0) {
            currentRow = 0;
            currentCol = 0;
            matrixSize = size;
            isFirstElement = true;
        }
        
        void printElement(double value) {
            if (isFirstElement) {
                printf("%s[Matrix Result]\n", getCurrentTimestamp().c_str());
                isFirstElement = false;
            }
            
            if (currentCol == 0) {
                printf("%s| ", getCurrentTimestamp().c_str());
            }
            
            printf("%8.4f ", value);
            currentCol++;
            
            if (currentCol >= matrixSize) {
                printf("|\n");
                currentCol = 0;
                currentRow++;
                
                if (currentRow >= matrixSize) {
                    printf("%s[End Matrix]\n", getCurrentTimestamp().c_str());
                    reset();
                }
            }
        }
        
        bool isActive() const {
            return matrixSize > 0;
        }
    };
    
    static MatrixPrintState matrixState;
}

extern "C" {

void printFloat(double value) {
    if (isBenchmarkMarker(value)) {
        // 如果是基准测试标记，重置矩阵状态并处理标记
        matrixState.reset();
        benchmarkTracker.handleValue(value);
    } else if (isExactInteger(value) && value >= 0 && value <= 100 && matrixState.isActive()) {
        // 只有在矩阵状态激活时才将整数视为矩阵大小标记
        matrixState.reset(static_cast<int>(value));
    } else {
        // 普通数值，可能是矩阵元素或单独的数字
        if (matrixState.isActive()) {
            matrixState.printElement(value);
        } else {
            printf("%s%g\n", getCurrentTimestamp().c_str(), value);
            fflush(stdout);
        }
    }
}

void printString(const char* str) {
    printf("%s%s\n", getCurrentTimestamp().c_str(), str);
    fflush(stdout);
}

void printInt(int64_t value) {
    printf("%s%lld\n", getCurrentTimestamp().c_str(), value);
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