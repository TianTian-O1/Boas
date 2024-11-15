#ifndef MLIR_TO_LLVM_H
#define MLIR_TO_LLVM_H

#include <string>
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/BasicBlock.h"

namespace matrix {

class MLIRToLLVM {
public:
    static std::string convertToLLVM(const std::vector<std::vector<float>>& matrixA,
                                   const std::vector<std::vector<float>>& matrixB);

    static std::string convertToLLVM(const std::string& mlirInput);

private:
    static llvm::Value* createMatrix(llvm::IRBuilder<>& builder,
                                   llvm::Module& module,
                                   const std::vector<float>& values,
                                   const std::vector<int>& shape);

    static llvm::Value* createMatrixMultiplication(llvm::IRBuilder<>& builder,
                                                llvm::Value* matA,
                                                llvm::Value* matB,
                                                const std::vector<int>& shapeA,
                                                const std::vector<int>& shapeB);

    static std::vector<int> broadcastShapes(const std::vector<int>& shapeA,
                                         const std::vector<int>& shapeB);

    static void createPrintMatrix(llvm::IRBuilder<>& builder,
                               llvm::Module& module,
                               llvm::Value* matrix,
                               const std::vector<int>& shape);
};

} // namespace matrix

#endif // MLIR_TO_LLVM_H