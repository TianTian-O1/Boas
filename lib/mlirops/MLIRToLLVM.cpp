#include "mlirops/MLIRToLLVM.h"
#include <sstream>
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Type.h"

namespace matrix {

std::string MLIRToLLVM::convertToLLVM(const std::string& mlirInput) {
    llvm::LLVMContext context;
    llvm::Module module("matrix_module", context);
    llvm::IRBuilder<> builder(context);

    // Create main function
    llvm::FunctionType* mainType = llvm::FunctionType::get(
        builder.getInt32Ty(), false);
    llvm::Function* mainFunc = llvm::Function::Create(
        mainType, llvm::Function::ExternalLinkage, "main", module);

    // Create entry basic block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", mainFunc);
    builder.SetInsertPoint(entry);

    // Create matrices A and B with shapes
    std::vector<float> valuesA = {1.0f, 2.0f, 3.0f, 4.0f};  // [[1, 2],
                                                            //  [3, 4]]
    std::vector<float> valuesB = {5.0f, 6.0f, 7.0f, 8.0f};  // [[5, 6],
                                                            //  [7, 8]]
    std::vector<int> shapeA = {2, 2};
    std::vector<int> shapeB = {2, 2};

    llvm::Value* matA = createMatrix(builder, module, valuesA, shapeA);
    llvm::Value* matB = createMatrix(builder, module, valuesB, shapeB);

    // Perform matrix multiplication
    llvm::Value* matC = createMatrixMultiplication(builder, matA, matB, shapeA, shapeB);

    // Calculate the shape of the result matrix (m,k) × (k,n) -> (m,n)
    std::vector<int> shapeC = {shapeA[0], shapeB[1]};  // Result shape is (2,2)

    // Print result
    createPrintMatrix(builder, module, matC, shapeC);

    // Return 0
    builder.CreateRet(builder.getInt32(0));

    // Verify and return
    std::string output;
    llvm::raw_string_ostream os(output);
    module.print(os, nullptr);
    return output;
}

llvm::Value* MLIRToLLVM::createMatrix(llvm::IRBuilder<>& builder,
                                     llvm::Module& module,
                                     const std::vector<float>& values,
                                     const std::vector<int>& shape) {
    int size = shape[0] * shape[1];
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), size);
    llvm::Value* matrix = builder.CreateAlloca(arrayTy, nullptr, "matrix");

    for (int i = 0; i < size; i++) {
        llvm::Value* idx = builder.getInt32(i);
        builder.CreateStore(
            llvm::ConstantFP::get(builder.getFloatTy(), values[i]),
            builder.CreateGEP(arrayTy, matrix, {builder.getInt32(0), idx}));
    }

    return matrix;
}

llvm::Value* MLIRToLLVM::createMatrixMultiplication(llvm::IRBuilder<>& builder,
                                                   llvm::Value* matA,
                                                   llvm::Value* matB,
                                                   const std::vector<int>& shapeA,
                                                   const std::vector<int>& shapeB) {
    int m = shapeA[0];  // A的行数
    int k = shapeA[1];  // A的列数 = B的行数
    int n = shapeB[1];  // B的列数

    // 创建结果矩阵 (m x n)
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), m * n);
    llvm::Value* result = builder.CreateAlloca(arrayTy, nullptr, "result");

    // 实现矩阵乘法
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            llvm::Value* sum = llvm::ConstantFP::get(builder.getFloatTy(), 0.0);
            
            for (int p = 0; p < k; ++p) {
                // 获取A矩阵元素
                int aIdx = i * k + p;
                llvm::Value* aVal = builder.CreateLoad(builder.getFloatTy(),
                    builder.CreateGEP(arrayTy, matA, 
                        {builder.getInt32(0), builder.getInt32(aIdx)}));

                // 获取B矩阵元素
                int bIdx = p * n + j;
                llvm::Value* bVal = builder.CreateLoad(builder.getFloatTy(),
                    builder.CreateGEP(arrayTy, matB,
                        {builder.getInt32(0), builder.getInt32(bIdx)}));

                // 相乘并累加
                llvm::Value* prod = builder.CreateFMul(aVal, bVal);
                sum = builder.CreateFAdd(sum, prod);
            }

            // 存储结果
            int resultIdx = i * n + j;
            builder.CreateStore(sum, 
                builder.CreateGEP(arrayTy, result,
                    {builder.getInt32(0), builder.getInt32(resultIdx)}));
        }
    }

    return result;
}

// 辅助函数：计算广播后的形状
std::vector<int> MLIRToLLVM::broadcastShapes(const std::vector<int>& shapeA, 
                                            const std::vector<int>& shapeB) {
    std::vector<int> result;
    result.resize(2);
    result[0] = std::max(shapeA[0], shapeB[0]);
    result[1] = std::max(shapeA[1], shapeB[1]);
    return result;
}

void MLIRToLLVM::createPrintMatrix(llvm::IRBuilder<>& builder,
                                  llvm::Module& module,
                                  llvm::Value* matrix,
                                  const std::vector<int>& shape) {
    // 创建 printf 函数声明
    llvm::Type* i8Ty = builder.getInt8Ty();
    llvm::Type* i8PtrTy = llvm::PointerType::get(i8Ty, 0);
    llvm::FunctionType* printfType = llvm::FunctionType::get(
        builder.getInt32Ty(), {i8PtrTy}, true);
    llvm::FunctionCallee printfFunc = module.getOrInsertFunction("printf", printfType);

    // 创建格式字符串
    std::string formatStr = "Matrix:\n";
    for (int i = 0; i < shape[0]; i++) {
        formatStr += "[";
        for (int j = 0; j < shape[1]; j++) {
            formatStr += "%.0f";
            if (j < shape[1] - 1) formatStr += ", ";
        }
        formatStr += "]\n";
    }

    llvm::Value* formatStrVal = builder.CreateGlobalString(formatStr, "format");

    // 加载和转换矩阵元素
    std::vector<llvm::Value*> printArgs = {formatStrVal};
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), shape[0] * shape[1]);
    
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            int idx = i * shape[1] + j;
            llvm::Value* val = builder.CreateLoad(builder.getFloatTy(),
                builder.CreateGEP(arrayTy, matrix,
                    {builder.getInt32(0), builder.getInt32(idx)}));
            printArgs.push_back(builder.CreateFPExt(val, builder.getDoubleTy()));
        }
    }

    // 调用 printf
    builder.CreateCall(printfFunc, printArgs);
}

} // namespace matrix