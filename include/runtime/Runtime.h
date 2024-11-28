#ifndef MATRIX_RUNTIME_H
#define MATRIX_RUNTIME_H

#include <cstdint>

extern "C" {

// Print a floating point number
void printFloat(double value);

// Print a string
void printString(const char* str);

// Print an integer
void printInt(int64_t value);

// Get current time in microseconds
double system_time_usec();

// Generate a random number between 0 and 1
double generate_random();

// Print a matrix
struct MemRefDescriptor {
    double* allocated;
    double* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

void printMemrefF64(int64_t rank, void* ptr);

} // extern "C"

#endif // MATRIX_RUNTIME_H 