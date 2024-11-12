#ifndef MATRIX_DEBUG_H
#define MATRIX_DEBUG_H

#include <iostream>
#include <string>
#include <cstdarg>

namespace matrix {

enum DebugLevel {
    DEBUG_NONE = 0,
    DEBUG_LEXER = 1,
    DEBUG_PARSER = 2,
    DEBUG_AST = 4,
    DEBUG_MLIR = 8,
    DEBUG_LLVM = 16,
    DEBUG_ALL = 31
};

class Debug {
public:
    static void setLevel(int level) { debugLevel = level; }
    static bool isEnabled(DebugLevel level) { return (debugLevel & level) != 0; }
    
    template<typename... Args>
    static void log(DebugLevel level, const char* fmt, Args... args) {
        if (isEnabled(level)) {
            printf("[%s] ", getPrefix(level));
            printf("%s", fmt);
            printf("\n");
        }
    }
    
    template<typename... Args>
    static void logf(DebugLevel level, const char* fmt, Args... args) {
        if (isEnabled(level)) {
            printf("[%s] ", getPrefix(level));
            printf(fmt, args...);
            printf("\n");
        }
    }

private:
    static int debugLevel;
    static const char* getPrefix(DebugLevel level) {
        switch(level) {
            case DEBUG_LEXER: return "LEXER";
            case DEBUG_PARSER: return "PARSER";
            case DEBUG_AST: return "AST";
            case DEBUG_MLIR: return "MLIR";
            case DEBUG_LLVM: return "LLVM";
            default: return "DEBUG";
        }
    }
};

} // namespace matrix

#endif // MATRIX_DEBUG_H
