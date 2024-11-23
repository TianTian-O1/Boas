#include "mlirops/TensorMemoryPool.h"

namespace matrix {

mlir::Value TensorMemoryPool::allocate(mlir::OpBuilder& builder,
                                      const std::vector<int64_t>& shape,
                                      mlir::Type elementType) {
    auto& pool = instance();
    
    // 1. 尝试查找可重用的块
    for (auto& block : pool.blocks) {
        if (!block.in_use && 
            block.elementType == elementType && 
            pool.shapesMatch(block.shape, shape)) {
            block.in_use = true;
            return block.memref;
        }
    }
    
    // 2. 如果没有找到合适的块,创建新的
    auto memRefType = mlir::MemRefType::get(shape, elementType);
    auto newMemRef = builder.create<mlir::memref::AllocOp>(
        builder.getUnknownLoc(),
        memRefType
    );
    
    // 3. 如果池未满,添加到池中
    if (pool.blocks.size() < MAX_BLOCKS) {
        pool.blocks.emplace_back(newMemRef, shape, elementType);
    }
    
    return newMemRef;
}

void TensorMemoryPool::deallocate(mlir::Value memref) {
    auto& pool = instance();
    for (auto& block : pool.blocks) {
        if (block.memref == memref) {
            block.in_use = false;
            return;
        }
    }
}

void TensorMemoryPool::clear() {
    auto& pool = instance();
    pool.blocks.clear();
}

bool TensorMemoryPool::shapesMatch(const std::vector<int64_t>& a,
                                 const std::vector<int64_t>& b) const {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

} // namespace matrix 