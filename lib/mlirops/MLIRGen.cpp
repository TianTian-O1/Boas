
#include "mlirops/MLIRGen.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Region.h"

namespace matrix {


MLIRGen::MLIRGen() {
    context = std::make_unique<mlir::MLIRContext>();
    
    // 加载必要的方言
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::linalg::LinalgDialect>();
    
    // 启用操作验证
    context->loadAllAvailableDialects();
    
    builder = std::make_unique<mlir::OpBuilder>(context.get());
}

// 其他代码保持不变...
mlir::FloatType MLIRGen::getF64Type() {
    return builder->getF64Type();
}

mlir::Value MLIRGen::createConstantF64(double value) {
    return builder->create<mlir::arith::ConstantFloatOp>(
        builder->getUnknownLoc(),
        llvm::APFloat(value),
        getF64Type());
}

mlir::Value MLIRGen::generateNumberMLIR(const NumberExprAST* number) {
    return createConstantF64(number->getValue());
}

mlir::Value MLIRGen::generateMLIRForNode(const ExprAST* node) {
    if (!node) {
        llvm::errs() << "[DEBUG] Node is null\n";
        return nullptr;
    }
    
    llvm::errs() << "[DEBUG] Processing node of kind: " << node->getKind() << "\n";
    
    switch (node->getKind()) {
        case ExprAST::Binary: {
            llvm::errs() << "[DEBUG] Processing Binary node\n";
            auto* binaryExpr = static_cast<const BinaryExprAST*>(node);
            llvm::errs() << "[DEBUG] Binary operator: " << binaryExpr->getOp() << "\n";
            
            // Handle matmul operation
            if (auto* matmulExpr = dynamic_cast<const MatmulExprAST*>(node)) {
                llvm::errs() << "[DEBUG] Found matmul operation\n";
                return generateMLIRForMatmul(matmulExpr);
            }
            
            // 检查是否是赋值操作
            if (binaryExpr->getOp() == "=") {
                llvm::errs() << "[DEBUG] Found assignment operation\n";
                auto* lhs = binaryExpr->getLHS();
                if (lhs->getKind() == ExprAST::Variable) {
                    auto* lhsVar = static_cast<const VariableExprAST*>(lhs);
                    auto* rhs = binaryExpr->getRHS();
                    auto rhsValue = generateMLIRForNode(rhs);
                    if (!rhsValue) return nullptr;
                    symbolTable[lhsVar->getName()] = rhsValue;
                    return rhsValue;
                }
            }
            
            return generateMLIRForBinary(binaryExpr);
        }
        case ExprAST::Import:
            return generateMLIRForImport(static_cast<const ImportAST*>(node));
            
        case ExprAST::Function:
            llvm::errs() << "[DEBUG] Processing Function node\n";
            return generateMLIRForFunction(static_cast<const FunctionAST*>(node));
            
        case ExprAST::Number:
            llvm::errs() << "[DEBUG] Generating MLIR for Number\n";
            return generateNumberMLIR(static_cast<const NumberExprAST*>(node));
            
        case ExprAST::Variable:
            llvm::errs() << "[DEBUG] Generating MLIR for Variable\n";
            return generateMLIRForVariable(static_cast<const VariableExprAST*>(node));
            
        case ExprAST::Assignment:
            llvm::errs() << "[DEBUG] Generating MLIR for Assignment\n";
            return generateMLIRForAssignment(static_cast<const AssignmentExprAST*>(node));
            
        case ExprAST::Matmul:
            llvm::errs() << "[DEBUG] Processing Matmul node\n";
            return generateMLIRForMatmul(llvm::cast<MatmulExprAST>(node));
            
        case ExprAST::Print:
            llvm::errs() << "[DEBUG] Generating MLIR for Print\n";
            return generateMLIRForPrint(static_cast<const PrintExprAST*>(node));
            
        case ExprAST::TensorCreate:
            llvm::errs() << "[DEBUG] Generating MLIR for TensorCreate\n";
            return generateMLIRForTensorCreate(static_cast<const TensorCreateExprAST*>(node));
            
        default:
            llvm::errs() << "[DEBUG] Unknown node type: " << node->getKind() << "\n";
            return nullptr;
    }
}

mlir::Value MLIRGen::generateMLIRForMatmul(const MatmulExprAST* expr) {
    auto lhs = generateMLIRForNode(expr->getLHS());
    auto rhs = generateMLIRForNode(expr->getRHS());
    
    if (!lhs || !rhs) return nullptr;
    
    auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
    
    if (!lhsType || !rhsType) return nullptr;
    
    // 创建结果矩阵
    auto resultType = mlir::MemRefType::get(
        {lhsType.getShape()[0], rhsType.getShape()[1]}, 
        builder->getF64Type()
    );
    
    auto result = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(), 
        resultType
    );
    
    // 初始化结果矩阵为 0
    auto zero = builder->create<mlir::arith::ConstantOp>(
        builder->getUnknownLoc(),
        builder->getF64FloatAttr(0.0)
    );
    
    // 创建 matmul 操作
    builder->create<mlir::linalg::MatmulOp>(
        builder->getUnknownLoc(),
        mlir::ValueRange{lhs, rhs},
        mlir::ValueRange{result}
    );
    
    return result;
}

mlir::Value MLIRGen::generateMLIRForPrint(const PrintExprAST* print) {
    llvm::errs() << "[DEBUG] Starting generateMLIRForPrint\n";
    
    // 获取要打印的表达式
    auto* expr = print->getValue();
    if (!expr) {
        llvm::errs() << "[DEBUG] Print expression is null\n";
        return nullptr;
    }
    
    llvm::errs() << "[DEBUG] Print expression kind: " << expr->getKind() << "\n";
    
    // 生成表达式的 MLIR 值
    auto value = generateMLIRForNode(expr);
    if (!value) {
        llvm::errs() << "[DEBUG] Failed to generate MLIR for print expression\n";
        return nullptr;
    }
    
    llvm::errs() << "[DEBUG] Generated value type: " << value.getType() << "\n";
    
    // 检查值的类型
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!memrefType) {
        llvm::errs() << "[DEBUG] Print value is not a memref type\n";
        return nullptr;
    }
    
    // 创建到通用 memref 类型的转换
    auto castedValue = builder->create<mlir::memref::CastOp>(
        builder->getUnknownLoc(),
        mlir::UnrankedMemRefType::get(builder->getF64Type(), 0),
        value
    );
    
    llvm::errs() << "[DEBUG] Created cast operation\n";
    
    // 调用打印函数
    builder->create<mlir::func::CallOp>(
        builder->getUnknownLoc(),
        "printMemrefF64",
        mlir::TypeRange{},
        mlir::ValueRange{castedValue}
    );
    
    llvm::errs() << "[DEBUG] Created print call\n";
    
    return value;
}

mlir::Value MLIRGen::generateMLIRForTensorCreate(const TensorCreateExprAST* expr) {
    // Cast to NumberExprAST to access getValue()
    auto rowsNum = static_cast<const NumberExprAST*>(expr->getRows());
    auto colsNum = static_cast<const NumberExprAST*>(expr->getCols());
    
    auto rows = static_cast<int64_t>(rowsNum->getValue());
    auto cols = static_cast<int64_t>(colsNum->getValue());
    
    // 创建 memref 类型
    auto memrefType = mlir::MemRefType::get({rows, cols}, builder->getF64Type());
    
    // 分配内存
    auto result = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(),
        memrefType
    );
    
    // 初始化值
    const auto& values = expr->getValues(); // Use reference instead of copying
    int idx = 0;
    
    // 遍历所有值并存储到 memref 中
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 获取当前值并转换为 NumberExprAST
            auto numberExpr = static_cast<const NumberExprAST*>(values[idx++].get());
            auto val = builder->create<mlir::arith::ConstantOp>(
                builder->getUnknownLoc(),
                builder->getF64FloatAttr(numberExpr->getValue())
            );
            
            // Create index values
            auto iVal = builder->create<mlir::arith::ConstantIndexOp>(
                builder->getUnknownLoc(), 
                i
            );
            auto jVal = builder->create<mlir::arith::ConstantIndexOp>(
                builder->getUnknownLoc(), 
                j
            );
            
            // Convert indices to ValueRange
            mlir::SmallVector<mlir::Value, 2> indices{iVal, jVal};
            
            // Create store operation
            builder->create<mlir::memref::StoreOp>(
                builder->getUnknownLoc(),
                val,
                result,
                indices
            );
        }
    }
    
    return result;
}

// Helper methods
mlir::MemRefType MLIRGen::getMemRefType(int64_t rows, int64_t cols) {
    return mlir::MemRefType::get({rows, cols}, getF64Type());
}

mlir::Value MLIRGen::createConstantIndex(int64_t value) {
    return builder->create<mlir::arith::ConstantIndexOp>(
        builder->getUnknownLoc(), 
        value
    );
}

mlir::ModuleOp MLIRGen::generateMLIR(const std::vector<std::unique_ptr<ExprAST>>& ast) {
    llvm::errs() << "[DEBUG] Starting MLIR generation\n";
    
    // Create module
    mlir::OpBuilder::InsertionGuard insertGuard(*builder);
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    
    // Register dialects
    auto *ctx = module.getContext();
    ctx->loadDialect<mlir::func::FuncDialect>();
    ctx->loadDialect<mlir::arith::ArithDialect>();
    ctx->loadDialect<mlir::memref::MemRefDialect>();
    ctx->loadDialect<mlir::linalg::LinalgDialect>();
    ctx->loadDialect<mlir::tensor::TensorDialect>();
    
    builder->setInsertionPointToStart(module.getBody());
    
    // Create printMemrefF64 declaration
    auto printMemrefType = mlir::FunctionType::get(
        ctx,
        {mlir::UnrankedMemRefType::get(builder->getF64Type(), 0)},
        {}
    );
    
    auto printFunc = mlir::func::FuncOp::create(
        builder->getUnknownLoc(),
        "printMemrefF64",
        printMemrefType
    );
    printFunc.setPrivate();
    module.push_back(printFunc);
    
    // Create main function (只创建一次)
    auto mainType = mlir::FunctionType::get(ctx, {}, {builder->getI32Type()});
    auto mainFunc = mlir::func::FuncOp::create(
        builder->getUnknownLoc(),
        "main",
        mainType
    );
    
    auto *entryBlock = mainFunc.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Process AST nodes
    for (const auto& node : ast) {
        if (!generateMLIRForNode(node.get())) {
            return nullptr;
        }
    }
    
    // Add return
    auto returnValue = builder->create<mlir::arith::ConstantIntOp>(
        builder->getUnknownLoc(),
        0,
        32
    );
    
    builder->create<mlir::func::ReturnOp>(
        builder->getUnknownLoc(),
        mlir::ValueRange{returnValue}
    );
    
    module.push_back(mainFunc);
    
    if (failed(mlir::verify(module))) {
        llvm::errs() << "[DEBUG] Module verification failed\n";
        module.dump();
        return nullptr;
    }
    
    return module;
}
std::string MLIRGen::getMLIRString(mlir::ModuleOp module) {
    std::string output;
    llvm::raw_string_ostream os(output);
    module.print(os);
    return output;
}

mlir::Value MLIRGen::generateMLIRForVariable(const VariableExprAST* expr) {
    auto it = symbolTable.find(expr->getName());
    if (it != symbolTable.end()) {
        return it->second;
    }
    
    llvm::errs() << "[DEBUG] Variable " << expr->getName() << " not found in symbol table\n";
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForAssignment(const AssignmentExprAST* expr) {
    llvm::errs() << "[DEBUG] Generating MLIR for Assignment\n";
    
    std::string varName = expr->getName();
    auto value = generateMLIRForNode(expr->getValue());
    
    if (!value) {
        llvm::errs() << "[DEBUG] Failed to generate value for assignment\n";
        return nullptr;
    }
    
    symbolTable[varName] = value;
    return value;
}




mlir::Value MLIRGen::generateMLIRForFunction(const FunctionAST* expr) {
    llvm::errs() << "[DEBUG] Generating MLIR for Function\n";
    
    // 获取函数体
    auto& body = expr->getBody();
    
    // 直接处理函数体里面的语句
    mlir::Value lastValue = nullptr;
    for (const auto& node : body) {
        lastValue = generateMLIRForNode(node.get());
        if (!lastValue) {
            llvm::errs() << "[DEBUG] Failed to generate MLIR for function body node\n";
            return nullptr;
        }
        llvm::errs() << "[DEBUG] Successfully generated MLIR for function body node\n";
    }
    
    return lastValue;
}

mlir::Value MLIRGen::generateMLIRForBinary(const BinaryExprAST* expr) {
    llvm::errs() << "[DEBUG] Processing Binary node\n";
    llvm::errs() << "[DEBUG] Binary operator: '" << expr->getOp() << "'\n";
    
    if (expr->getOp() == "matmul") {
        llvm::errs() << "[DEBUG] Found matmul operation\n";
        
        // 获取左右操作数
        auto* lhs = static_cast<const VariableExprAST*>(expr->getLHS());
        auto* rhs = static_cast<const VariableExprAST*>(expr->getRHS());
        
        llvm::errs() << "[DEBUG] Matmul operands: " << lhs->getName() << " * " << rhs->getName() << "\n";
        
        // 从符号表中获取变量
        auto lhsValue = symbolTable[lhs->getName()];
        auto rhsValue = symbolTable[rhs->getName()];
        
        if (!lhsValue || !rhsValue) {
            llvm::errs() << "[DEBUG] Failed to find operands in symbol table\n";
            return nullptr;
        }
        
        // 创建 MatmulExprAST 对象
        auto matmulExpr = std::make_unique<MatmulExprAST>(
            std::make_unique<VariableExprAST>(lhs->getName()),
            std::make_unique<VariableExprAST>(rhs->getName())
        );
        
        // 生成矩阵乘法操作
        auto result = generateMLIRForMatmul(matmulExpr.get());
        
        // 检查结果类型
        if (!result || !mlir::isa<mlir::MemRefType>(result.getType())) {
            llvm::errs() << "[DEBUG] Matmul result must be memref type\n";
            return nullptr;
        }
        
        // 如果这是一个赋值操作的一部分，结果会被上层处理
        return result;
    }
    
    return nullptr;
}


mlir::Value MLIRGen::generateMLIRForImport(const ImportAST* expr) {
    llvm::errs() << "[DEBUG] Processing Import node\n";
    
    // 获取导入的模块名称
    std::string moduleName = expr->getModuleName();
    llvm::errs() << "[DEBUG] Importing module: " << moduleName << "\n";
    
    // 如果是 tensor 模块，加载相关的 dialect
    if (moduleName == "tensor") {
        // 这些 dialect 其实已经在 generateMLIR 中加载过了，这里可以省略
        // 但为了安全起见，我们还是保留这些加载
        context->loadDialect<mlir::tensor::TensorDialect>();
        context->loadDialect<mlir::linalg::LinalgDialect>();
        context->loadDialect<mlir::arith::ArithDialect>();
        context->loadDialect<mlir::func::FuncDialect>();
        context->loadDialect<mlir::memref::MemRefDialect>();

        // 返回一个 dummy value，因为导入操作本身不需要产生值
        return builder->create<mlir::arith::ConstantOp>(
            builder->getUnknownLoc(),
            builder->getI32IntegerAttr(0)
        );
    }
    
    llvm::errs() << "[DEBUG] Unknown module: " << moduleName << "\n";
    return nullptr;
}



} // namespace matrix
