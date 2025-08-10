#pragma once
#include <vector>
#include <memory>
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Location.h"

namespace matrix {

class TensorMemoryPool {
private:
    struct MemoryBlock {
        mlir::Value memref;
        std::vector<int64_t> shape;
        bool in_use;
        mlir::Type elementType;
        
        MemoryBlock(mlir::Value m, const std::vector<int64_t>& s, mlir::Type t)
            : memref(m), shape(s), in_use(true), elementType(t) {}
    };
    
    std::vector<MemoryBlock> blocks;
    static constexpr size_t MAX_BLOCKS = 64;
    
    static TensorMemoryPool& instance() {
        static TensorMemoryPool pool;
        return pool;
    }
    
    bool shapesMatch(const std::vector<int64_t>& a, 
                    const std::vector<int64_t>& b) const;

public:
    static mlir::Value allocate(mlir::OpBuilder& builder, 
                               const std::vector<int64_t>& shape,
                               mlir::Type elementType);
    static void deallocate(mlir::Value memref);
    static void clear();
};

} // namespace matrix 