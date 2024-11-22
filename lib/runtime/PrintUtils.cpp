#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <random>
#include <thread>

namespace {
    struct PrintState {
        int currentRow = 0;
        int currentCol = 0;
        int totalRows = 0;
        int totalCols = 0;
        bool waitingForDimensions = true;
        bool gotRows = false;
        
        void reset() {
            currentRow = 0;
            currentCol = 0;
            totalRows = 0;
            totalCols = 0;
            waitingForDimensions = true;
            gotRows = false;
        }
    };
    
    PrintState state;
}

extern "C" {

// Helper function to write a formatted double
void writeDouble(double value) {
    printf("%.4f", value);
    fflush(stdout);
}

void writeString(const char* str) {
    printf("%s", str);
    fflush(stdout);
}

void writeNewline() {
    printf("\n");
    fflush(stdout);
}

void printFloat(double value) {
    // Handle timestamp
    if (value > 1e6) {
        auto milliseconds = static_cast<int64_t>(value);
        auto timePoint = std::chrono::system_clock::time_point(
            std::chrono::milliseconds(milliseconds)
        );
        auto time = std::chrono::system_clock::to_time_t(timePoint);
        auto localTime = *std::localtime(&time);
        auto ms = milliseconds % 1000;
        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &localTime);
        printf("%s.%03ld\n", buffer, ms);
        fflush(stdout);
        return;
    }

    // Handle matrix printing
    if (state.waitingForDimensions) {
        if (!state.gotRows) {
            // First number is rows
            state.totalRows = static_cast<int>(value);
            state.gotRows = true;
            return;
        } else {
            // Second number is cols
            state.totalCols = static_cast<int>(value);
            writeString("Matrix [");
            printf("%d x %d", state.totalRows, state.totalCols);
            writeString("]:\n[");
            writeNewline();
            state.waitingForDimensions = false;
            return;
        }
    }

    // Print matrix elements
    if (state.currentCol == 0) {
        writeString("  [");  // Start new row
        writeDouble(value);
    } else {
        writeString(", ");
        writeDouble(value);
    }

    state.currentCol++;
    
    // Handle end of row
    if (state.currentCol >= state.totalCols) {
        writeString("]");  // Close row brackets
        if (state.currentRow < state.totalRows - 1) {
            writeString(",");  // Add comma if not last row
        }
        writeNewline();
        
        state.currentCol = 0;
        state.currentRow++;
        
        // Check if matrix is complete
        if (state.currentRow >= state.totalRows) {
            writeString("]");
            writeNewline();
            state.reset();  // Reset state for next matrix
        }
    }
}

void printString(const char* str) {
    printf("%s\n", str);
    fflush(stdout);
}

void printInt(int64_t value) {
    printf("%ld\n", value);
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

}