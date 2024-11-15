#include "mlirops/MLIRToLLVM.h"
#include <sstream>
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Type.h"
#include <regex>

namespace matrix {

std::string MLIRToLLVM::convertToLLVM(const std::vector<std::vector<float>>& matrixA,
                                     const std::vector<std::vector<float>>& matrixB) {
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

    // Convert input matrices to flat vectors
    std::vector<float> valuesA;
    std::vector<float> valuesB;
    std::vector<int> shapeA = {(int)matrixA.size(), (int)matrixA[0].size()};
    std::vector<int> shapeB = {(int)matrixB.size(), (int)matrixB[0].size()};

    // Flatten matrix A
    for (const auto& row : matrixA) {
        valuesA.insert(valuesA.end(), row.begin(), row.end());
    }

    // Flatten matrix B
    for (const auto& row : matrixB) {
        valuesB.insert(valuesB.end(), row.begin(), row.end());
    }

    // Create LLVM matrices
    llvm::Value* matA = createMatrix(builder, module, valuesA, shapeA);
    llvm::Value* matB = createMatrix(builder, module, valuesB, shapeB);

    // Perform matrix multiplication
    llvm::Value* matC = createMatrixMultiplication(builder, matA, matB, shapeA, shapeB);

    // Calculate result shape
    std::vector<int> shapeC = {shapeA[0], shapeB[1]};

    // Print result
    createPrintMatrix(builder, module, matC, shapeC);

    // Return 0
    builder.CreateRet(builder.getInt32(0));

    // Generate output
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
    int m = shapeA[0];
    int k = shapeA[1];
    int n = shapeB[1];

    // 创建结果矩阵
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), m * n);
    llvm::Value* result = builder.CreateAlloca(arrayTy, nullptr, "result");

    // 定义分块大小
    const int BLOCK_SIZE = 32;
    const int VECTOR_SIZE = 4;  // AVX支持4个float并行计算

    // 创建向量类型
    llvm::VectorType* vectorTy = llvm::VectorType::get(builder.getFloatTy(), VECTOR_SIZE, false);

    // 外层循环：按块遍历
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                // 内层循环：在块内计算
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, m); ii++) {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, n); jj += VECTOR_SIZE) {
                        // 初始化向量累加器
                        llvm::Value* sumVec = llvm::Constant::getNullValue(vectorTy);

                        // 向量化内层循环
                        for (int pp = p; pp < std::min(p + BLOCK_SIZE, k); pp++) {
                            // 加载A矩阵元素
                            llvm::Value* aVal = builder.CreateLoad(builder.getFloatTy(),
                                builder.CreateGEP(arrayTy, matA,
                                    {builder.getInt32(0),
                                     builder.getInt32(ii * k + pp)}));

                            // 广播A矩阵元素到向量
                            llvm::Value* aVec = builder.CreateVectorSplat(VECTOR_SIZE, aVal);

                            // 加载B矩阵向量
                            std::vector<llvm::Value*> bElements;
                            for (int v = 0; v < VECTOR_SIZE && jj + v < n; v++) {
                                bElements.push_back(
                                    builder.CreateLoad(builder.getFloatTy(),
                                        builder.CreateGEP(arrayTy, matB,
                                            {builder.getInt32(0),
                                             builder.getInt32(pp * n + jj + v)}))
                                );
                            }
                            llvm::Value* bVec = builder.CreateVectorSplat(VECTOR_SIZE, bElements[0]);
                            for (int v = 1; v < VECTOR_SIZE && jj + v < n; v++) {
                                bVec = builder.CreateInsertElement(bVec, bElements[v], 
                                    builder.getInt32(v));
                            }

                            // 向量乘法和累加
                            llvm::Value* prodVec = builder.CreateFMul(aVec, bVec);
                            sumVec = builder.CreateFAdd(sumVec, prodVec);
                        }

                        // 存储结果向量
                        for (int v = 0; v < VECTOR_SIZE && jj + v < n; v++) {
                            llvm::Value* sum = builder.CreateExtractElement(sumVec, 
                                builder.getInt32(v));
                            builder.CreateStore(sum,
                                builder.CreateGEP(arrayTy, result,
                                    {builder.getInt32(0),
                                     builder.getInt32(ii * n + jj + v)}));
                        }
                    }
                }
            }
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

/*static*/ std::string MLIRToLLVM::convertToLLVM(const std::string& mlirInput) {
    std::vector<std::vector<float>> matrixA;
    std::vector<std::vector<float>> matrixB;
    
    // 使用正则表达式提取矩阵值
    std::regex matrix_pattern("\"tensor\\.from_elements\"\\(\\)\\s*\\(\\{([^}]+)\\}\\)");
    std::smatch matches;
    std::string::const_iterator searchStart(mlirInput.cbegin());
    
    // 找到第一个矩阵 (A)
    if (std::regex_search(searchStart, mlirInput.cend(), matches, matrix_pattern)) {
        std::string elements = matches[1].str();
        std::stringstream ss(elements);
        std::vector<float> values;
        float val;
        
        // 解析所有数字
        while (ss >> val) {
            values.push_back(val);
            // 跳过逗号和空格
            char c;
            while (ss >> c && (c == ',' || c == ' ' || c == '\n')) {
                continue;
            }
            if (ss.good()) {
                ss.unget();
            }
        }
        
        // 转换为2x2矩阵
        if (values.size() >= 4) {
            matrixA.push_back({values[0], values[1]});
            matrixA.push_back({values[2], values[3]});
        }
        
        searchStart = matches.suffix().first;
    }
    
    // 找到第二个矩阵 (B)
    if (std::regex_search(searchStart, mlirInput.cend(), matches, matrix_pattern)) {
        std::string elements = matches[1].str();
        std::stringstream ss(elements);
        std::vector<float> values;
        float val;
        
        while (ss >> val) {
            values.push_back(val);
            char c;
            while (ss >> c && (c == ',' || c == ' ' || c == '\n')) {
                continue;
            }
            if (ss.good()) {
                ss.unget();
            }
        }
        
        if (values.size() >= 4) {
            matrixB.push_back({values[0], values[1]});
            matrixB.push_back({values[2], values[3]});
        }
    }
    
    // 调用矩阵乘法实现
    return convertToLLVM(matrixA, matrixB);
}

} // namespace matrix