#include "runtime/Runtime.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

EXPORT void printFloat(double value) {
    std::cout << std::fixed << std::setprecision(6) << value << std::endl;
}

EXPORT void printString(const char* str) {
    std::cout << str << std::endl;
}

EXPORT void printInt(int64_t value) {
    std::cout << value << std::endl;
}

EXPORT double system_time_usec() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return static_cast<double>(micros.count());
}

EXPORT double generate_random() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}

EXPORT void printMemrefF64(int64_t rank, void* ptr) {
    if (!ptr) {
        std::cout << "null" << std::endl;
        return;
    }

    auto desc = static_cast<MemRefDescriptor*>(ptr);
    auto data = desc->aligned;
    auto sizes = desc->sizes;
    auto strides = desc->strides;

    if (rank == 2) {
        auto rows = sizes[0];
        auto cols = sizes[1];
        
        std::cout << "\nMatrix " << rows << "x" << cols << ":" << std::endl;
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                auto idx = desc->offset + i * strides[0] + j * strides[1];
                std::cout << std::fixed << std::setprecision(6) << std::setw(12) 
                         << data[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    } else {
        std::cout << "Unsupported rank: " << rank << std::endl;
    }
}

} // extern "C"