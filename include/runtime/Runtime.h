#ifndef MATRIX_RUNTIME_H
#define MATRIX_RUNTIME_H

#include <cstdint>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Print a floating point number
EXPORT void printFloat(double value);

// Print a string
EXPORT void printString(const char* str);

// Print an integer
EXPORT void printInt(int64_t value);

// Get current time in microseconds
EXPORT double system_time_usec();

// Generate a random number between 0 and 1
EXPORT double generate_random();

// Print a matrix
struct MemRefDescriptor {
    double* allocated;
    double* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

EXPORT void printMemrefF64(int64_t rank, void* ptr);

} // extern "C"

#endif // MATRIX_RUNTIME_H 