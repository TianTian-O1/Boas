// MLIRGenTensor.cpp - 张量操作相关实现
#include "mlirops/MLIRGen.h"
#include "mlirops/TensorMemoryPool.h"

namespace matrix {

mlir::Value MLIRGen::generateMLIRForTensor(const TensorExprAST* expr) {
    std::cerr << "[DEBUG] Generating Tensor operation\n";
    
    const auto& elements = expr->getElements();
    if (elements.empty()) {
        std::cerr << "Error: Empty tensor elements\n";
        return nullptr;
    }
    
    std::vector<mlir::Value> values;
    for (const auto& element : elements) {
        auto value = generateMLIRForNode(element.get());
        if (!value) {
            std::cerr << "Error: Failed to generate tensor element\n";
            return nullptr;
        }
        values.push_back(value);
    }
    
    auto elementType = builder->getF64Type();
    auto shape = mlir::MemRefType::get({static_cast<int64_t>(values.size())}, elementType);
    
    auto result = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(),
        shape
    );
    
    for (size_t i = 0; i < values.size(); ++i) {
        auto idx = builder->create<mlir::arith::ConstantIndexOp>(
            builder->getUnknownLoc(),
            i
        );
        
        builder->create<mlir::memref::StoreOp>(
            builder->getUnknownLoc(),
            values[i],
            result,
            mlir::ValueRange{idx}
        );
    }
    
    return result;
}

mlir::Value MLIRGen::generateMLIRForTensorCreate(const TensorCreateExprAST* expr) {
    llvm::errs() << "Creating tensor...\n";
    
    auto rowsNum = static_cast<const NumberExprAST*>(expr->getRows())->getValue();
    auto colsNum = static_cast<const NumberExprAST*>(expr->getCols())->getValue();
    
    auto resultType = mlir::MemRefType::get(
        {static_cast<int64_t>(rowsNum), static_cast<int64_t>(colsNum)},
        builder->getF64Type()
    );
    
    auto result = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(),
        resultType
    );
    
    const auto& values = expr->getValues();
    int idx = 0;
    
    for (int i = 0; i < rowsNum; i++) {
        for (int j = 0; j < colsNum; j++) {
            auto val = builder->create<mlir::arith::ConstantOp>(
                builder->getUnknownLoc(),
                builder->getF64FloatAttr(
                    static_cast<const NumberExprAST*>(values[idx++].get())->getValue()
                )
            );
            
            auto iVal = createConstantIndex(i);
            auto jVal = createConstantIndex(j);
            
            builder->create<mlir::memref::StoreOp>(
                builder->getUnknownLoc(),
                val,
                result,
                mlir::ValueRange{iVal, jVal}
            );
        }
    }
    
    llvm::errs() << "Tensor created successfully\n";
    return result;
}


mlir::Value MLIRGen::generateMLIRForTensorRandom(const TensorRandomExprAST* expr) {
    std::cerr << "[DEBUG] Generating random tensor using optimized approach\n";
    
    auto* rowsNum = static_cast<const NumberExprAST*>(expr->getRows());
    auto* colsNum = static_cast<const NumberExprAST*>(expr->getCols());
    
    int64_t rows = static_cast<int64_t>(rowsNum->getValue());
    int64_t cols = static_cast<int64_t>(colsNum->getValue());
    
    if (rows <= 0 || cols <= 0 || rows > 2048 || cols > 2048) {
        std::cerr << "Invalid dimensions for random tensor: " << rows << "x" << cols << "\n";
        return nullptr;
    }

    auto loc = builder->getUnknownLoc();
    
    // 1. 分配结果内存
    std::vector<int64_t> shape = {rows, cols};
    auto memrefType = mlir::MemRefType::get(shape, builder->getF64Type());
    auto result = builder->create<mlir::memref::AllocOp>(loc, memrefType);
    
    // 2. 创建一维缓冲区用于存储随机数
    const int64_t vectorSize = 8;  // SIMD向量大小
    const int64_t batchSize = std::min(int64_t(32), rows * cols);  // 批处理大小
    auto bufferType = mlir::MemRefType::get({batchSize}, builder->getF64Type());
    auto randomBuffer = builder->create<mlir::memref::AllocOp>(loc, bufferType);
    
    // 3. 声明外部随机数生成函数
    mlir::func::FuncOp genRandomFunc;
    if (!(genRandomFunc = module.lookupSymbol<mlir::func::FuncOp>("generate_random"))) {
        auto funcType = mlir::FunctionType::get(
            builder->getContext(),
            {},
            {builder->getF64Type()}
        );
        genRandomFunc = mlir::func::FuncOp::create(loc, "generate_random", funcType);
        mlir::SymbolTable symbolTable(module);
        symbolTable.insert(genRandomFunc);
    }

    // 4. 先生成一批随机数
    auto fillLoop = builder->create<mlir::scf::ForOp>(
        loc,
        createConstantIndex(0),
        createConstantIndex(batchSize),
        createConstantIndex(1),
        mlir::ValueRange(),
        [&](mlir::OpBuilder& fillBuilder, 
            mlir::Location fillLoc, 
            mlir::Value fillIdx, 
            mlir::ValueRange fillArgs) {
              // 生成随机数并存入缓冲区
              auto randVal = fillBuilder.create<mlir::func::CallOp>(
                  fillLoc,
                  genRandomFunc,
                  mlir::ValueRange{}
              ).getResult(0);
              
              fillBuilder.create<mlir::memref::StoreOp>(
                  fillLoc,
                  randVal,
                  randomBuffer,
                  mlir::ValueRange{fillIdx}
              );
              
              fillBuilder.create<mlir::scf::YieldOp>(fillLoc);
        });
        
    // 5. 使用嵌套循环填充结果矩阵
    int64_t totalElements = rows * cols;
    auto mainLoop = builder->create<mlir::scf::ForOp>(
        loc,
        createConstantIndex(0),
        createConstantIndex(totalElements),
        createConstantIndex(1),
        mlir::ValueRange(),
        [&](mlir::OpBuilder& mainBuilder, 
            mlir::Location mainLoc, 
            mlir::Value idx, 
            mlir::ValueRange mainArgs) {
              // 计算当前元素的行列索引
              auto i = mainBuilder.create<mlir::arith::DivUIOp>(
                  mainLoc,
                  idx,
                  createConstantIndex(cols)
              ).getResult();
              
              auto j = mainBuilder.create<mlir::arith::RemUIOp>(
                  mainLoc,
                  idx,
                  createConstantIndex(cols)
              ).getResult();
              
              // 从随机缓冲区读取值
              auto bufferIdx = mainBuilder.create<mlir::arith::RemUIOp>(
                  mainLoc,
                  idx,
                  createConstantIndex(batchSize)
              ).getResult();
              
              auto randVal = mainBuilder.create<mlir::memref::LoadOp>(
                  mainLoc,
                  randomBuffer,
                  mlir::ValueRange{bufferIdx}
              ).getResult();
              
              // 存储到结果矩阵
              mainBuilder.create<mlir::memref::StoreOp>(
                  mainLoc,
                  randVal,
                  result,
                  mlir::ValueRange{i, j}
              );
              
              mainBuilder.create<mlir::scf::YieldOp>(mainLoc);
        });
        
    // 6. 释放临时缓冲区
    builder->create<mlir::memref::DeallocOp>(loc, randomBuffer);
    
    // 7. 添加优化属性
    if (auto* defOp = result.getOperation()) {
        optimizeMemoryAccess(defOp);
        addVectorizationAttributes(defOp);
        addParallelizationAttributes(defOp);
        addTilingAttributes(defOp, 32);
    }

    std::cerr << "[DEBUG] Optimized tensor generation complete\n";
    return result;
}


} // namespace matrix