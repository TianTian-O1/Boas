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
        return isExactInteger(value) && value >= 1.0 && value <= 10.0 && 
               value == std::floor(value);  // Must be whole number
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
}

extern "C" {

void printFloat(double value) {
    benchmarkTracker.handleValue(value);
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