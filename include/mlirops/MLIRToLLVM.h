#ifndef MLIR_TO_LLVM_H
#define MLIR_TO_LLVM_H

#include <string>
#include <vector>
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
    struct MatrixDimensions {
        int rows;
        int cols;
        std::vector<float> data;
    };

    static std::vector<float> flattenMatrix(const std::vector<std::vector<float>>& matrix);

    static llvm::Value* createMatrix(llvm::IRBuilder<>& builder,
                                   llvm::Module& module,
                                   const std::vector<float>& values,
                                   const std::vector<int>& shape);

    static llvm::Constant* createMatrixConstant(llvm::IRBuilder<>& builder,
                                              const std::vector<float>& values,
                                              int rows,
                                              int cols);

    static llvm::Value* createBroadcastMatrix(llvm::IRBuilder<>& builder,
                                            llvm::Module& module,
                                            llvm::Value* srcMat,
                                            const std::vector<int>& srcShape,
                                            const std::vector<int>& targetShape);

    static llvm::Value* createMatrixMultiplication(llvm::IRBuilder<>& builder,
                                                llvm::Module& module,
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

    static MatrixDimensions parseMatrixDimensions(const std::string& elements,
                                                const std::string& type);

    static std::vector<std::vector<float>> parseMatrix(const std::string& elements,
                                                     int rows,
                                                     int cols);

    static std::vector<float> broadcastValues(const std::vector<float>& values,
                                            int sourceRows,
                                            int sourceCols,
                                            int targetRows,
                                            int targetCols);
};

} // namespace matrix

#endif // MLIR_TO_LLVM_H