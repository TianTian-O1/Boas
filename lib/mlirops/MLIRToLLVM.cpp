#include "mlirops/MLIRToLLVM.h"
#include <sstream>
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Type.h"
#include <regex>
#include <iostream>

namespace matrix {

std::vector<std::vector<float>> broadcastMatrix(const std::vector<std::vector<float>>& matrix,
                                               int targetRows,
                                               int targetCols) {
    int sourceRows = matrix.size();
    int sourceCols = matrix[0].size();

    // **Check for broadcasting compatibility**
    if (sourceRows != targetRows && sourceRows != 1 && targetRows != 1) {
        throw std::runtime_error("Cannot broadcast matrices: incompatible number of rows (" +
                                 std::to_string(sourceRows) + " vs. " + std::to_string(targetRows) + ")");
    }
    if (sourceCols != targetCols && sourceCols != 1 && targetCols != 1) {
        throw std::runtime_error("Cannot broadcast matrices: incompatible number of columns (" +
                                 std::to_string(sourceCols) + " vs. " + std::to_string(targetCols) + ")");
    }

    std::vector<std::vector<float>> result(targetRows, std::vector<float>(targetCols));
    
    for (int i = 0; i < targetRows; i++) {
        int i_src = sourceRows == 1 ? 0 : i % sourceRows;
        for (int j = 0; j < targetCols; j++) {
            int j_src = sourceCols == 1 ? 0 : j % sourceCols;
            result[i][j] = matrix[i_src][j_src];
        }
    }
    return result;
}

std::string MLIRToLLVM::convertToLLVM(const std::vector<std::vector<float>>& matrixA,
                                     const std::vector<std::vector<float>>& matrixB) {
    std::cout << "[DEBUG] Starting LLVM IR generation\n";
    
    int rowsA = matrixA.size();
    int colsA = matrixA[0].size();
    int rowsB = matrixB.size();
    int colsB = matrixB[0].size();

    // **Check inner dimensions for matrix multiplication**
    if (colsA != rowsB) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication (" +
                                 std::to_string(colsA) + " vs. " + std::to_string(rowsB) + ")");
    }

    // **Determine target dimensions for broadcasting**
    int targetRows = std::max(rowsA, rowsB);
    int targetCols = std::max(colsA, colsB);

    // **Check broadcasting compatibility**
    if ((rowsA != targetRows && rowsA != 1) || (rowsB != targetRows && rowsB != 1)) {
        throw std::runtime_error("Cannot broadcast matrices: incompatible number of rows (" +
                                 std::to_string(rowsA) + " vs. " + std::to_string(rowsB) + ")");
    }
    if ((colsA != targetCols && colsA != 1) || (colsB != targetCols && colsB != 1)) {
        throw std::runtime_error("Cannot broadcast matrices: incompatible number of columns (" +
                                 std::to_string(colsA) + " vs. " + std::to_string(colsB) + ")");
    }

    // **Broadcast matrices**
    auto broadcastedA = broadcastMatrix(matrixA, targetRows, targetCols);
    auto broadcastedB = broadcastMatrix(matrixB, targetRows, targetCols);
    
    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> module = std::make_unique<llvm::Module>("matrix_module", *context);
    std::unique_ptr<llvm::IRBuilder<>> builder = std::make_unique<llvm::IRBuilder<>>(*context);
    
    // Get matrix dimensions
    rowsA = broadcastedA.size();
    colsA = broadcastedA[0].size();
    rowsB = broadcastedB.size();
    colsB = broadcastedB[0].size();
    
    // Create main function
    llvm::FunctionType* mainType = llvm::FunctionType::get(builder->getInt32Ty(), false);
    llvm::Function* mainFunc = llvm::Function::Create(mainType, 
                                                     llvm::Function::ExternalLinkage,
                                                     "main", 
                                                     *module);
    
    // Create entry basic block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(*context, "entry", mainFunc);
    builder->SetInsertPoint(entry);
    
    // Create global arrays for matrices
    std::vector<float> flatA = flattenMatrix(broadcastedA);
    std::vector<float> flatB = flattenMatrix(broadcastedB);
    
    llvm::ArrayType* arrayTypeA = llvm::ArrayType::get(builder->getFloatTy(), flatA.size());
    llvm::ArrayType* arrayTypeB = llvm::ArrayType::get(builder->getFloatTy(), flatB.size());
    
    std::vector<llvm::Constant*> constantsA;
    std::vector<llvm::Constant*> constantsB;
    
    for (float val : flatA) {
        constantsA.push_back(llvm::ConstantFP::get(builder->getFloatTy(), val));
    }
    for (float val : flatB) {
        constantsB.push_back(llvm::ConstantFP::get(builder->getFloatTy(), val));
    }
    
    llvm::Constant* initA = llvm::ConstantArray::get(arrayTypeA, constantsA);
    llvm::Constant* initB = llvm::ConstantArray::get(arrayTypeB, constantsB);
    
    llvm::GlobalVariable* globalA = new llvm::GlobalVariable(
        *module, arrayTypeA, true, llvm::GlobalValue::PrivateLinkage, initA, "matrixA");
    llvm::GlobalVariable* globalB = new llvm::GlobalVariable(
        *module, arrayTypeB, true, llvm::GlobalValue::PrivateLinkage, initB, "matrixB");
    
    // Create result matrix
    int resultSize = rowsA * colsB;
    llvm::ArrayType* resultType = llvm::ArrayType::get(builder->getFloatTy(), resultSize);
    llvm::AllocaInst* result = builder->CreateAlloca(resultType, nullptr, "result");
    
    // Generate matrix multiplication code
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            llvm::Value* sum = llvm::ConstantFP::get(builder->getFloatTy(), 0.0f);
            
            for (int k = 0; k < colsA; k++) {
                // Load A[i][k]
                llvm::Value* idxA = builder->getInt32(i * colsA + k);
                llvm::Value* ptrA = builder->CreateGEP(arrayTypeA, globalA, 
                    {builder->getInt32(0), idxA});
                llvm::Value* valA = builder->CreateLoad(builder->getFloatTy(), ptrA);
                
                // Load B[k][j]
                llvm::Value* idxB = builder->getInt32(k * colsB + j);
                llvm::Value* ptrB = builder->CreateGEP(arrayTypeB, globalB,
                    {builder->getInt32(0), idxB});
                llvm::Value* valB = builder->CreateLoad(builder->getFloatTy(), ptrB);
                
                // Multiply and add
                llvm::Value* prod = builder->CreateFMul(valA, valB);
                sum = builder->CreateFAdd(sum, prod);
            }
            
            // Store result
            llvm::Value* resultIdx = builder->getInt32(i * colsB + j);
            llvm::Value* resultPtr = builder->CreateGEP(resultType, result,
                {builder->getInt32(0), resultIdx});
            builder->CreateStore(sum, resultPtr);
            
            // Print the result (you'll need to implement a print function)
            // This is just a placeholder - you'll need to add proper printing functionality
        }
    }
    
    // Declare printf function
    llvm::FunctionType* printfType = llvm::FunctionType::get(
        builder->getInt32Ty(),
        {llvm::PointerType::get(builder->getInt8Ty(), 0)},
        true
    );
    llvm::FunctionCallee printfFunc = module->getOrInsertFunction("printf", printfType);

    // Create format string constant
    std::string formatStr = "Result matrix:\n[%.2f %.2f]\n[%.2f %.2f]\n";
    llvm::Constant* formatStrConst = builder->CreateGlobalString(formatStr, "format");
    llvm::Value* formatStrPtr = builder->CreatePointerCast(
        formatStrConst,
        llvm::PointerType::get(builder->getInt8Ty(), 0)
    );

    // Load result values
    std::vector<llvm::Value*> printfArgs;
    printfArgs.push_back(formatStrPtr);

    // Load and add each result value to printf arguments
    for (int i = 0; i < 4; i++) {
        llvm::Value* ptr = builder->CreateGEP(resultType, result,
            {builder->getInt32(0), builder->getInt32(i)});
        llvm::Value* val = builder->CreateLoad(builder->getFloatTy(), ptr);
        // Convert float to double for printf
        llvm::Value* doubleVal = builder->CreateFPExt(val, builder->getDoubleTy());
        printfArgs.push_back(doubleVal);
    }

    // Call printf
    builder->CreateCall(printfFunc, printfArgs);
    
    // Return 0 from main
    builder->CreateRet(builder->getInt32(0));
    
    // Generate LLVM IR string
    std::string output;
    llvm::raw_string_ostream os(output);
    module->print(os, nullptr);
    
    return output;
}

llvm::Value* MLIRToLLVM::createMatrix(llvm::IRBuilder<>& builder,
                                     llvm::Module& module,
                                     const std::vector<float>& values,
                                     const std::vector<int>& shape) {
    std::cout << "[DEBUG] Creating LLVM array with " << values.size() << " elements.\n";
    std::cout << "[DEBUG] Matrix shape: " << shape[0] << "x" << shape[1] << "\n";
    
    int size = shape[0] * shape[1];
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), size);
    llvm::Value* matrix = builder.CreateAlloca(arrayTy, nullptr, "matrix");
    
    // Initialize matrix elements
    for (int i = 0; i < size && i < values.size(); i++) {
        builder.CreateStore(
            llvm::ConstantFP::get(builder.getFloatTy(), values[i]),
            builder.CreateGEP(arrayTy, matrix, {builder.getInt32(0), builder.getInt32(i)})
        );
    }
    
    return matrix;
}

llvm::Value* MLIRToLLVM::createMatrixMultiplication(llvm::IRBuilder<>& builder,
                                                   llvm::Module& module,
                                                   llvm::Value* matA,
                                                   llvm::Value* matB,
                                                   const std::vector<int>& shapeA,
                                                   const std::vector<int>& shapeB) {
    // Validate matrix dimensions
    if (shapeA[1] != shapeB[0]) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication");
    }
    
    int M = shapeA[0];
    int N = shapeB[1];
    int K = shapeA[1];
    
    // Create result matrix
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), M * N);
    llvm::Value* result = builder.CreateAlloca(arrayTy, nullptr, "result");
    
    // Initialize result matrix to zero
    for (int i = 0; i < M * N; i++) {
        builder.CreateStore(
            llvm::ConstantFP::get(builder.getFloatTy(), 0.0f),
            builder.CreateGEP(arrayTy, result, {builder.getInt32(0), builder.getInt32(i)})
        );
    }
    
    // Matrix multiplication C[i,j] = sum(A[i,k] * B[k,j])
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            llvm::Value* sum = llvm::ConstantFP::get(builder.getFloatTy(), 0.0f);
            
            for (int k = 0; k < K; k++) {
                // Load A[i,k]
                llvm::Value* aVal = builder.CreateLoad(builder.getFloatTy(),
                    builder.CreateGEP(arrayTy, matA,
                        {builder.getInt32(0), builder.getInt32(i * K + k)}));
                
                // Load B[k,j]
                llvm::Value* bVal = builder.CreateLoad(builder.getFloatTy(),
                    builder.CreateGEP(arrayTy, matB,
                        {builder.getInt32(0), builder.getInt32(k * N + j)}));
                
                // Multiply and accumulate
                llvm::Value* prod = builder.CreateFMul(aVal, bVal);
                sum = builder.CreateFAdd(sum, prod);
            }
            
            // Store C[i,j]
            builder.CreateStore(sum,
                builder.CreateGEP(arrayTy, result,
                    {builder.getInt32(0), builder.getInt32(i * N + j)}));
        }
    }
    
    return result;
}

// 增强的广播形状计算函数
std::vector<int> MLIRToLLVM::broadcastShapes(const std::vector<int>& shapeA, 
                                            const std::vector<int>& shapeB) {
    // For matrix multiplication (M x K) * (K x N) = (M x N)
    if (shapeA[1] != shapeB[0]) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication");
    }
    return {shapeA[0], shapeB[1]};
}

// 创建广播矩阵的辅助函数
llvm::Value* MLIRToLLVM::createBroadcastMatrix(llvm::IRBuilder<>& builder,
                                              llvm::Module& module,
                                              llvm::Value* srcMat,
                                              const std::vector<int>& srcShape,
                                              const std::vector<int>& targetShape) {
    int targetSize = targetShape[0] * targetShape[1];
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), targetSize);
    llvm::Value* result = builder.CreateAlloca(arrayTy, nullptr, "broadcast_matrix");
    
    for (int i = 0; i < targetShape[0]; i++) {
        for (int j = 0; j < targetShape[1]; j++) {
            // 修正广播索引计算
            int srcI = (srcShape[0] == 1) ? 0 : i;
            int srcJ = (srcShape[1] == 1) ? 0 : j;
            
            // 确保索引不会越界
            srcI = std::min(srcI, srcShape[0] - 1);
            srcJ = std::min(srcJ, srcShape[1] - 1);
            
            // 加载源矩阵元素
            llvm::Value* srcIdx = builder.CreateGEP(arrayTy, srcMat,
                {builder.getInt32(0),
                 builder.getInt32(srcI * srcShape[1] + srcJ)});
            llvm::Value* srcVal = builder.CreateLoad(builder.getFloatTy(), srcIdx);
            
            // 存储到目标矩阵
            llvm::Value* dstIdx = builder.CreateGEP(arrayTy, result,
                {builder.getInt32(0),
                 builder.getInt32(i * targetShape[1] + j)});
            builder.CreateStore(srcVal, dstIdx);
        }
    }
    
    return result;
}

void MLIRToLLVM::createPrintMatrix(llvm::IRBuilder<>& builder,
                                  llvm::Module& module,
                                  llvm::Value* matrix,
                                  const std::vector<int>& shape) {
    // 创建格式字串
    std::string formatStr = "Matrix:\n";
    for (int i = 0; i < shape[0]; i++) {
        formatStr += "[";
        for (int j = 0; j < shape[1]; j++) {
            formatStr += "%.2f";
            if (j < shape[1] - 1) formatStr += ", ";
        }
        formatStr += "]\n";
    }
    
    // 创建 printf 函数声明
    llvm::Type* i8Ty = builder.getInt8Ty();
    llvm::Type* i8PtrTy = llvm::PointerType::get(i8Ty, 0);
    llvm::FunctionType* printfType = llvm::FunctionType::get(
        builder.getInt32Ty(), {i8PtrTy}, true);
    llvm::FunctionCallee printfFunc = module.getOrInsertFunction("printf", printfType);

    llvm::Value* formatStrVal = builder.CreateGlobalString(formatStr, "format");

    // 加载和转换矩阵元素
    std::vector<llvm::Value*> printArgs = {formatStrVal};
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), shape[0] * shape[1]);
    
    // 按照正确的顺序加载矩阵元素
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
    std::cout << "[DEBUG] Starting MLIR to LLVM conversion\n";
    std::cout << "[DEBUG] MLIR Input:\n" << mlirInput << "\n";
    
    std::vector<std::vector<float>> matrixA;
    std::vector<std::vector<float>> matrixB;
    
    // Updated regex pattern to match tensor.from_elements format with multiline elements
    std::regex tensor_pattern(R"(%([A-Z])\s*=\s*"tensor\.from_elements"\(\)\s*\(\{\s*((?:.|\s)*?)\s*\}\)\s*\{type\s*=\s*tensor<(\d+)x(\d+)xf64>\})");
    std::smatch matches;
    std::string::const_iterator searchStart(mlirInput.cbegin());
    
    std::cout << "[DEBUG] Attempting to parse first matrix\n";
    // Parse first matrix (A)
    if (std::regex_search(searchStart, mlirInput.cend(), matches, tensor_pattern)) {
        std::cout << "[DEBUG] Found first matrix pattern\n";
        std::string matrixName = matches[1];
        std::string elementsStr = matches[2];
        int rows = std::stoi(matches[3]);
        int cols = std::stoi(matches[4]);
        
        std::cout << "[DEBUG] Matrix Name: " << matrixName << "\n";
        std::cout << "[DEBUG] Elements String:\n" << elementsStr << "\n";
        std::cout << "[DEBUG] Rows: " << rows << ", Cols: " << cols << "\n";
        
        // Parse matrix elements
        std::vector<float> values;
        std::regex number_pattern(R"([\d\.]+)");
        auto numbers_begin = std::sregex_iterator(elementsStr.begin(), elementsStr.end(), number_pattern);
        auto numbers_end = std::sregex_iterator();
        
        for (auto i = numbers_begin; i != numbers_end; ++i) {
            values.push_back(std::stof((*i).str()));
            std::cout << "[DEBUG] Parsed value: " << values.back() << "\n";
        }
        
        // Create matrixA
        matrixA = std::vector<std::vector<float>>(rows, std::vector<float>(cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixA[i][j] = values[i * cols + j];
            }
        }
        
        searchStart = matches.suffix().first;
    } else {
        std::cout << "[DEBUG] Failed to match first matrix pattern\n";
        throw std::runtime_error("Failed to parse first matrix");
    }
    
    std::cout << "[DEBUG] Attempting to parse second matrix\n";
    // Parse second matrix (B)
    if (std::regex_search(searchStart, mlirInput.cend(), matches, tensor_pattern)) {
        std::cout << "[DEBUG] Found second matrix pattern\n";
        std::string matrixName = matches[1];
        std::string elementsStr = matches[2];
        int rows = std::stoi(matches[3]);
        int cols = std::stoi(matches[4]);
        
        std::cout << "[DEBUG] Matrix Name: " << matrixName << "\n";
        std::cout << "[DEBUG] Elements String:\n" << elementsStr << "\n";
        std::cout << "[DEBUG] Rows: " << rows << ", Cols: " << cols << "\n";
        
        // Parse matrix elements
        std::vector<float> values;
        std::regex number_pattern(R"([\d\.]+)");
        auto numbers_begin = std::sregex_iterator(elementsStr.begin(), elementsStr.end(), number_pattern);
        auto numbers_end = std::sregex_iterator();
        
        for (auto i = numbers_begin; i != numbers_end; ++i) {
            values.push_back(std::stof((*i).str()));
            std::cout << "[DEBUG] Parsed value: " << values.back() << "\n";
        }
        
        // Create matrixB
        matrixB = std::vector<std::vector<float>>(rows, std::vector<float>(cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixB[i][j] = values[i * cols + j];
            }
        }
        
        searchStart = matches.suffix().first;
    } else {
        std::cout << "[DEBUG] Failed to match second matrix pattern\n";
        throw std::runtime_error("Failed to parse second matrix");
    }
    
    // Now generate LLVM IR using the matrices
    return convertToLLVM(matrixA, matrixB);
}

std::vector<float> MLIRToLLVM::flattenMatrix(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> result;
    for (const auto& row : matrix) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

llvm::Constant* MLIRToLLVM::createMatrixConstant(llvm::IRBuilder<>& builder,
                                                const std::vector<float>& values,
                                                int rows,
                                                int cols) {
    std::vector<llvm::Constant*> constants;
    int expectedSize = rows * cols;
    for (int i = 0; i < expectedSize; i++) {
        float val = (i < values.size()) ? values[i] : 0.0f;
        constants.push_back(llvm::ConstantFP::get(builder.getFloatTy(), val));
    }
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), expectedSize);
    return llvm::ConstantArray::get(arrayTy, constants);
}

std::vector<std::vector<float>> MLIRToLLVM::parseMatrix(const std::string& elementsStr, int rows, int cols) {
    std::vector<float> values;

    // Regular expression to extract numbers
    std::regex number_pattern(R"([\d\.]+)");
    auto numbers_begin = std::sregex_iterator(elementsStr.begin(), elementsStr.end(), number_pattern);
    auto numbers_end = std::sregex_iterator();

    for (auto i = numbers_begin; i != numbers_end; ++i) {
        values.push_back(std::stof((*i).str()));
    }

    // **Check if we have the correct number of elements**
    int expectedSize = rows * cols;
    if (values.size() != expectedSize) {
        throw std::runtime_error("Number of elements does not match the specified matrix dimensions.");
    }

    // Initialize the matrix
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    int idx = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = values[idx++];
        }
    }

    return matrix;
}

MLIRToLLVM::MatrixDimensions MLIRToLLVM::parseMatrixDimensions(const std::string& elements,
                                                              const std::string& type) {
    MatrixDimensions dims;
    // Parse type string like "tensor<1x2xf64>" to get dimensions
    std::smatch match;
    std::regex typeRegex(R"(tensor<(\d+)x(\d+)x)");
    if (std::regex_search(type, match, typeRegex)) {
        dims.rows = std::stoi(match[1]);
        dims.cols = std::stoi(match[2]);
    }
    
    // Parse elements
    std::stringstream ss(elements);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) {
            dims.data.push_back(std::stof(item));
        }
    }
    return dims;
}

std::vector<float> MLIRToLLVM::broadcastValues(const std::vector<float>& values,
                                              int sourceRows,
                                              int sourceCols,
                                              int targetRows,
                                              int targetCols) {
    std::vector<float> result(targetRows * targetCols);
    
    for (int i = 0; i < targetRows; i++) {
        for (int j = 0; j < targetCols; j++) {
            int sourceRow = i % sourceRows;
            int sourceCol = j % sourceCols;
            result[i * targetCols + j] = values[sourceRow * sourceCols + sourceCol];
        }
    }
    
    return result;
}

std::vector<std::vector<float>> initializeMatrix(const std::vector<float>& values, 
                                               int rows, 
                                               int cols) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols, 0.0f));
    int idx = 0;
    for (int i = 0; i < rows && idx < values.size(); i++) {
        for (int j = 0; j < cols && idx < values.size(); j++) {
            matrix[i][j] = values[idx++];
        }
    }
    return matrix;
}

llvm::Constant* createMatrixConstant(llvm::IRBuilder<>& builder,
                                   const std::vector<std::vector<float>>& matrix) {
    std::vector<llvm::Constant*> constants;
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            constants.push_back(llvm::ConstantFP::get(builder.getFloatTy(), matrix[i][j]));
        }
    }
    
    llvm::ArrayType* arrayTy = llvm::ArrayType::get(builder.getFloatTy(), rows * cols);
    return llvm::ConstantArray::get(arrayTy, constants);
}

} // namespace matrix