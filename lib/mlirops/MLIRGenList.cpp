#include "mlirops/MLIRGen.h"

namespace matrix {

mlir::Value MLIRGen::generateList(const ListExprAST* list) {
    if (!list) return nullptr;
    
    auto loc = builder->getUnknownLoc();
    std::vector<mlir::Value> elements;
    
    // 生成所有列表元素的 MLIR 值
    for (const auto& element : list->getElements()) {
        if (auto value = this->generate(element.get())) {
            elements.push_back(value);
        }
    }
    
    return createList(elements);
}

mlir::Value MLIRGen::createList(const std::vector<mlir::Value>& elements) {
    auto loc = builder->getUnknownLoc();
    
    // 创建动态大小的 memref 来存储列表元素
    auto listType = mlir::MemRefType::get(
        {static_cast<int64_t>(elements.size())}, 
        builder->getF64Type()
    );
    
    auto listAlloc = builder->create<mlir::memref::AllocOp>(loc, listType);
    
    // 存储列表元素
    for (size_t i = 0; i < elements.size(); ++i) {
        auto idx = createConstantIndex(i);
        builder->create<mlir::memref::StoreOp>(
            loc, elements[i], listAlloc, idx
        );
    }
    
    return listAlloc;
}

mlir::Value MLIRGen::getListElement(mlir::Value list, mlir::Value index) {
    auto loc = builder->getUnknownLoc();
    
    // 确保列表是一维 memref
    auto listType = list.getType().dyn_cast<mlir::MemRefType>();
    if (!listType || listType.getRank() != 1) {
        std::cerr << "Error: List must be a 1D memref\n";
        return nullptr;
    }
    
    // 使用 ValueRange 来传递单个索引
    return builder->create<mlir::memref::LoadOp>(
        loc, 
        list, 
        mlir::ValueRange{index}
    );
}

mlir::Value MLIRGen::setListElement(mlir::Value list, 
                                   mlir::Value index, 
                                   mlir::Value value) {
    auto loc = builder->getUnknownLoc();
    
    // 确保列表是一维 memref
    auto listType = list.getType().dyn_cast<mlir::MemRefType>();
    if (!listType || listType.getRank() != 1) {
        std::cerr << "Error: List must be a 1D memref\n";
        return nullptr;
    }
    
    // 使用 ValueRange 来传递单个索引
    builder->create<mlir::memref::StoreOp>(
        loc, 
        value, 
        list, 
        mlir::ValueRange{index}
    );
    return list;
}

mlir::Value MLIRGen::generateListIndex(const ListIndexExprAST* expr) {
    if (!expr) return nullptr;
    
    auto loc = builder->getUnknownLoc();
    
    // 生成列表和索引的 MLIR 值
    auto list = generate(expr->getList());
    auto index = generate(expr->getIndex());
    
    if (!list || !index) {
        std::cerr << "Failed to generate list or index\n";
        return nullptr;
    }
    
    // 确保列表是一维 memref
    auto listType = list.getType().dyn_cast<mlir::MemRefType>();
    if (!listType || listType.getRank() != 1) {
        std::cerr << "Error: List must be a 1D memref\n";
        return nullptr;
    }
    
    // 如果索引是浮点数，先转换为整数
    if (index.getType().isF64()) {
        // 先转换为 i32
        index = builder->create<mlir::arith::FPToSIOp>(
            loc,
            builder->getI32Type(),
            index
        );
        // 再转换为 index 类型
        index = builder->create<mlir::arith::IndexCastOp>(
            loc,
            builder->getIndexType(),
            index
        );
    }
    // 如果已经是整数类型但不是 index，直接转换为 index
    else if (!index.getType().isIndex()) {
        index = builder->create<mlir::arith::IndexCastOp>(
            loc,
            builder->getIndexType(),
            index
        );
    }
    
    // 使用单个索引访问一维 memref
    return builder->create<mlir::memref::LoadOp>(loc, list, mlir::ValueRange{index});
}

} // namespace matrix 