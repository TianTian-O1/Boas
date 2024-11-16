#include "mlirops/MLIRGen.h"
#include <sstream>
#include <set>

namespace matrix {

std::string MLIRGen::generateMLIR(const std::vector<std::unique_ptr<ExprAST>>& ast) {
    std::stringstream ss;
    ss << "module {\n";
    ss << "  \"builtin.module\"() ({\n";
    
    // Handle imports first
    for (const auto& node : ast) {
        if (auto* import = dynamic_cast<const ImportAST*>(node.get())) {
            ss << "    " << generateMLIRForImport(import);
        }
    }
    
    // Generate main function
    ss << "    \"func.func\"() ({\n";
    ss << "    func @main() {\n";
    
    // Handle function body
    for (const auto& node : ast) {
        if (auto* func = dynamic_cast<const FunctionAST*>(node.get())) {
            for (const auto& expr : func->getBody()) {
                if (auto* assign = dynamic_cast<const AssignmentExprAST*>(expr.get())) {
                    ss << generateMLIRForAssignment(assign) << "\n";
                } else if (auto* print = dynamic_cast<const PrintExprAST*>(expr.get())) {
                    ss << "      \"tensor.print\"(" << generateExpression(print->getValue()) 
                       << ") : (tensor<2x2xf64>)\n";
                }
            }
        }
    }
    
    ss << "    }\n";
    ss << "    }) {sym_name = \"main\", function_type = () -> (), sym_visibility = \"public\"}\n";
    ss << "  }) {}\n";
    ss << "}\n";
    
    return ss.str();
}

std::string MLIRGen::generateMLIRForNode(const ExprAST* node) {
    if (auto number = dynamic_cast<const NumberExprAST*>(node)) {
        return generateNumberMLIR(number);
    } else if (auto variable = dynamic_cast<const VariableExprAST*>(node)) {
        return generateMLIRForVariable(variable);
    } else if (auto tensor = dynamic_cast<const TensorExprAST*>(node)) {
        return generateMLIRForTensor(tensor);
    } else if (auto import = dynamic_cast<const ImportAST*>(node)) {
        return generateMLIRForImport(import);
    } else if (auto function = dynamic_cast<const FunctionAST*>(node)) {
        return generateMLIRForFunction(function);
    } else if (auto assignment = dynamic_cast<const AssignmentExprAST*>(node)) {
        return generateMLIRForAssignment(assignment);
    } else if (auto call = dynamic_cast<const CallExprAST*>(node)) {
        return generateMLIRForCall(call);
    } else if (auto print = dynamic_cast<const PrintExprAST*>(node)) {
        return generateMLIRForPrint(print);
    }
    std::cerr << "Unhandled AST node type in MLIR generation\n";
    return "";
}

std::string MLIRGen::generateMLIRForImport(const ImportAST* import) {
    return "  // Import: " + import->getModuleName() + "\n";
}

std::string MLIRGen::generateMLIRForFunction(const FunctionAST* function) {
    std::stringstream ss;
    ss << "  func @" << function->getName() << "() {\n";
    
    // 处理函数体
    for (const auto& expr : function->getBody()) {
        ss << generateMLIRForNode(expr.get()) << "\n";
    }
    
    ss << "  }";
    return ss.str();
}

std::string MLIRGen::generateMLIRForAssignment(const AssignmentExprAST* assignment) {
    std::string varName = assignment->getName();
    std::string rhs;
    std::stringstream ss;
    
    const ExprAST* rhsExpr = assignment->getRHS();
    if (auto* tensor = dynamic_cast<const TensorCreateExprAST*>(rhsExpr)) {
        ss << "    %" << varName << " = " << generateTensorCreate(tensor);
    } else if (auto* matmul = dynamic_cast<const MatmulExprAST*>(rhsExpr)) {
        auto* lhs = dynamic_cast<const VariableExprAST*>(matmul->getLHS());
        auto* rhs = dynamic_cast<const VariableExprAST*>(matmul->getRHS());
        
        if (!lhs || !rhs) {
            throw std::runtime_error("Matrix multiplication operands must be variables");
        }
        
        ss << "    %" << varName << " = \"linalg.matmul\"(%" << lhs->getName() 
           << ", %" << rhs->getName() << ") : "
           << "(tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>";
    }
    
    return ss.str();
}

std::string MLIRGen::generateMLIRForArray(const ArrayExprAST* array) {
    std::stringstream ss;
    const auto& elements = array->getElements();
    
    ss << "[\n";
    for (size_t i = 0; i < elements.size(); ++i) {
        ss << "      ";
        if (auto number = dynamic_cast<const NumberExprAST*>(elements[i].get())) {
            ss << number->getValue();
        }
        if (i < elements.size() - 1) ss << ",\n";
    }
    ss << "\n    ] : tensor<" << elements.size() << "xf64>";
    
    return ss.str();
}

std::string MLIRGen::generateMLIRForTensor(const TensorExprAST* tensor) {
    std::stringstream ss;
    ss << "      \"tensor.from_elements\"() ({\n";
    const auto& elements = tensor->getElements();
    
    for (size_t i = 0; i < elements.size(); ++i) {
        ss << "        ";
        if (auto array = dynamic_cast<const ArrayExprAST*>(elements[i].get())) {
            const auto& row = array->getElements();
            for (size_t j = 0; j < row.size(); ++j) {
                if (auto number = dynamic_cast<const NumberExprAST*>(row[j].get())) {
                    ss << number->getValue();
                    if (j < row.size() - 1) ss << ", ";
                }
            }
        }
        if (i < elements.size() - 1) ss << ",\n";
    }
    
    ss << "\n      }) {type = tensor<" << elements.size() << "x2xf64>}";
    return ss.str();
}


std::string MLIRGen::generateMLIRForVariable(const VariableExprAST* variable) {
    return "%" + variable->getName();
}

std::string MLIRGen::generateNumberMLIR(const NumberExprAST* number) {
    return std::to_string(number->getValue());
}

std::string MLIRGen::generateMLIRForCall(const CallExprAST* call) {
    std::stringstream ss;
    if (call->getMemberName() == "matmul") {
        const auto& args = call->getArguments();
        if (args.size() == 2) {
            ss << "      \"linalg.matmul\"(";
            ss << generateMLIRForNode(args[0].get()) << ", ";
            ss << generateMLIRForNode(args[1].get()) << ") : ";
            ss << "(tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>";
        }
    }
    return ss.str();
}

std::string MLIRGen::generateMLIRForPrint(const PrintExprAST* print) {
    std::stringstream ss;
    if (auto* var = dynamic_cast<const VariableExprAST*>(print->getValue())) {
        ss << "      \"tensor.print\"(%" << var->getName() << ") : (tensor<2x2xf64>)";
    }
    return ss.str();
}

std::string MLIRGen::generateExpression(const ExprAST* expr) {
    if (auto* tensor = dynamic_cast<const TensorCreateExprAST*>(expr)) {
        return generateTensorCreate(tensor);
    } else if (auto* matmul = dynamic_cast<const MatmulExprAST*>(expr)) {
        return generateMLIRForMatmul(matmul);
    } else if (auto* var = dynamic_cast<const VariableExprAST*>(expr)) {
        return generateMLIRForVariable(var);
    }
    return ""; // Handle other cases as needed
}

std::string MLIRGen::generateTensorCreate(const TensorCreateExprAST* expr) {
    // Get dimensions
    auto rows = dynamic_cast<const NumberExprAST*>(expr->getRows());
    auto cols = dynamic_cast<const NumberExprAST*>(expr->getCols());
    
    if (!rows || !cols) {
        throw std::runtime_error("Tensor dimensions must be numeric constants");
    }
    
    int numRows = static_cast<int>(rows->getValue());
    int numCols = static_cast<int>(cols->getValue());
    
    // Build the elements string
    std::string elements;
    const auto& values = expr->getValues();
    for (size_t i = 0; i < values.size(); i++) {
        if (auto num = dynamic_cast<const NumberExprAST*>(values[i].get())) {
            if (i > 0) elements += ", ";
            elements += std::to_string(num->getValue());
        } else {
            throw std::runtime_error("Tensor elements must be numeric constants");
        }
    }
    
    // Generate MLIR tensor creation
    std::stringstream ss;
    ss << "\"tensor.from_elements\"() ({" << elements << "}) "
       << "{type = tensor<" << numRows << "x" << numCols << "xf64>}";
    
    return ss.str();
}

std::string MLIRGen::generateMLIRForMatmul(const MatmulExprAST* matmul) {
    std::string lhs = generateMLIRForNode(matmul->getLHS());
    std::string rhs = generateMLIRForNode(matmul->getRHS());
    
    std::string temp = getNextTemp();
    std::stringstream ss;
    ss << temp << " = \"linalg.matmul\"(" << lhs << ", " << rhs << ") : "
       << "(tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>";
    
    return ss.str();
}

} // namespace matrix
