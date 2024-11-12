#ifndef MATRIX_MLIR_GEN_H
#define MATRIX_MLIR_GEN_H

#include "frontend/AST.h"
#include <string>
#include <vector>
#include <memory>

namespace matrix {

class MLIRGen {
public:
    MLIRGen() = default;
    std::string generateMLIR(const std::vector<std::unique_ptr<ExprAST>>& ast);

private:
    std::string generateMLIRForNode(const ExprAST* node);
    std::string generateMLIRForImport(const ImportAST* import);
    std::string generateMLIRForFunction(const FunctionAST* function);
    std::string generateMLIRForAssignment(const AssignmentExprAST* assignment);
    std::string generateMLIRForArray(const ArrayExprAST* array);
    std::string generateMLIRForTensor(const TensorExprAST* tensor);
    std::string generateMLIRForMatmul(const MatmulExprAST* matmul);
    std::string generateMLIRForVariable(const VariableExprAST* variable);
    std::string generateNumberMLIR(const NumberExprAST* number);
    std::string generateMLIRForCall(const CallExprAST* call);
    std::string generateMLIRForPrint(const PrintExprAST* print);
    
    // 辅助方法
    std::string getNextTemp() {
        return "%" + std::to_string(temp_counter_++);
    }
    
private:
    int temp_counter_ = 0;
};

} // namespace matrix

#endif // MATRIX_MLIR_GEN_H
