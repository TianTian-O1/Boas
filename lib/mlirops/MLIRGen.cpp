#include "mlirops/MLIRGen.h"
#include <sstream>
#include <set>

namespace matrix {

std::string MLIRGen::generateMLIR(const std::vector<std::unique_ptr<ExprAST>>& ast) {
    std::stringstream ss;
    ss << "module {\n";
    // Add required dialect registrations
    ss << "  \"builtin.module\"() ({\n";
    ss << "    \"func.func\"() ({\n";
    
    for (const auto& node : ast) {
        if (auto import = dynamic_cast<const ImportAST*>(node.get())) {
            ss << "      // Import: " << import->getModuleName() << "\n";
        } else {
            ss << generateMLIRForNode(node.get()) << "\n";
        }
    }
    
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
    std::stringstream ss;
    ss << "    %" << assignment->getName() << " = ";
    
    const ExprAST* value = assignment->getValue();
    if (auto tensor = dynamic_cast<const TensorExprAST*>(value)) {
        ss << generateMLIRForTensor(tensor);
    } else if (auto call = dynamic_cast<const CallExprAST*>(value)) {
        ss << generateMLIRForCall(call);
    } else {
        ss << "// Unknown assignment value type";
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

std::string MLIRGen::generateMLIRForMatmul(const MatmulExprAST* matmul) {
    std::stringstream ss;
    auto lhs = dynamic_cast<const VariableExprAST*>(matmul->getLHS());
    auto rhs = dynamic_cast<const VariableExprAST*>(matmul->getRHS());
    
    ss << "linalg.matmul\n"
       << "    ins(%" << lhs->getName() << ", %" << rhs->getName() << " : tensor<2x2xf64>, tensor<2x2xf64>)\n"
       << "    outs(%result : tensor<2x2xf64>)\n"
       << "    -> tensor<2x2xf64>";
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
    ss << "      \"tensor.print\"(" << generateMLIRForNode(print->getValue()) << ") : (tensor<2x2xf64>)";
    return ss.str();
}

} // namespace matrix
