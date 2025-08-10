#include "frontend/Parser.h"
#include "frontend/ASTImpl.h"
#include "Debug.h"

namespace matrix {

int Parser::getTokPrecedence() {
    if (!isascii(current_token_.kind)) {
        return -1;
    }
    
    auto it = binop_precedence_.find(std::string(1, current_token_.kind));
    if (it == binop_precedence_.end()) {
        return -1;
    }
    return it->second;
}

std::unique_ptr<ExprAST> Parser::LogError(const std::string &str) {
    llvm::errs() << "Error: " << str << "\n";
    return nullptr;
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
    auto LHS = parsePrimary();
    if (!LHS) {
        return nullptr;
    }
    
    return parseBinOpRHS(0, std::move(LHS));
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
    switch (current_token_.kind) {
        case tok_left_paren:
            return parseParenExpr();
        case tok_identifier:
            return parseIdentifierExpr();
        case tok_number:
            return parseNumberExpr();
        default:
            return LogError("Unknown token when expecting an expression");
    }
}

std::unique_ptr<ExprAST> Parser::parseNumberExpr() {
    auto result = std::make_unique<NumberExprASTImpl>(std::stod(current_token_.value));
    getNextToken(); // consume the number
    return std::move(result);
}

std::unique_ptr<ExprAST> Parser::parseParenExpr() {
    getNextToken(); // eat (
    auto V = parseExpression();
    if (!V) {
        return nullptr;
    }
    
    if (current_token_.kind != tok_right_paren) {
        return LogError("Expected ')'");
    }
    getNextToken(); // eat )
    return V;
}

std::unique_ptr<ExprAST> Parser::parseIdentifierExpr() {
    std::string IdName = current_token_.value;
    getNextToken(); // eat identifier
    
    if (current_token_.kind == tok_dot) {  // 方法调用，如 tensor.to()
        getNextToken(); // eat .
        std::string methodName = current_token_.value;
        getNextToken(); // eat method name
        
        if (methodName == "to") {
            if (current_token_.kind != tok_left_paren) {
                return LogError("Expected '(' after 'to'");
            }
            getNextToken(); // eat (
            
            // 解析设备名称
            if (current_token_.kind != tok_string) {
                return LogError("Expected device name string in to()");
            }
            std::string deviceName = current_token_.value;
            getNextToken(); // eat device name
            
            if (current_token_.kind != tok_right_paren) {
                return LogError("Expected ')' after device name");
            }
            getNextToken(); // eat )
            
            return std::make_unique<DeviceTransferExprAST>(
                std::make_unique<VariableExprASTImpl>(IdName),
                deviceName
            );
        }
        // 其他方法调用的处理...
    }
    
    if (current_token_.kind != tok_left_paren) { // Simple variable ref
        return std::make_unique<VariableExprASTImpl>(IdName);
    }
    
    // Call
    getNextToken(); // eat (
    std::vector<std::unique_ptr<ExprAST>> Args;
    if (current_token_.kind != tok_right_paren) {
        while (1) {
            if (auto Arg = parseExpression()) {
                Args.push_back(std::move(Arg));
            } else {
                return nullptr;
            }
            
            if (current_token_.kind == tok_right_paren) {
                break;
            }
            
            if (current_token_.kind != tok_comma) {
                return LogError("Expected ')' or ',' in argument list");
            }
            getNextToken();
        }
    }
    getNextToken(); // eat )
    
    return std::make_unique<CallExprASTImpl>(IdName, std::move(Args));
}

std::unique_ptr<ExprAST> Parser::parseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS) {
    while (true) {
        int TokPrec = getTokPrecedence();
        
        if (TokPrec < ExprPrec) {
            return LHS;
        }
        
        std::string BinOp = std::string(1, current_token_.kind);
        getNextToken(); // eat binop
        
        auto RHS = parsePrimary();
        if (!RHS) {
            return nullptr;
        }
        
        int NextPrec = getTokPrecedence();
        if (TokPrec < NextPrec) {
            RHS = parseBinOpRHS(TokPrec + 1, std::move(RHS));
            if (!RHS) {
                return nullptr;
            }
        }
        
        LHS = std::make_unique<BinaryExprASTImpl>(std::move(LHS), std::move(RHS), BinOp);
    }
}

std::unique_ptr<ExprAST> Parser::parseTensorExpr() {
    if (current_token_.kind == tok_random) {
        return parseTensorRandomExpr();
    } else if (current_token_.kind == tok_create) {
        return parseTensorCreateExpr();
    }
    return nullptr;
}

std::unique_ptr<ExprAST> Parser::parseTensorRandomExpr() {
    if (current_token_.kind != tok_left_paren) {
        return LogError("Expected '(' in tensor.random");
    }
    getNextToken(); // eat (
    
    auto rows = parseExpression();
    if (!rows) {
        return nullptr;
    }
    
    if (current_token_.kind != tok_comma) {
        return LogError("Expected ',' after rows in tensor.random");
    }
    getNextToken(); // eat ,
    
    auto cols = parseExpression();
    if (!cols) {
        return nullptr;
    }
    
    if (current_token_.kind != tok_right_paren) {
        return LogError("Expected ')' after cols in tensor.random");
    }
    getNextToken(); // eat )
    
    return std::make_unique<TensorRandomExprASTImpl>(
        std::move(rows),
        std::move(cols)
    );
}

std::unique_ptr<ExprAST> Parser::parseTensorCreateExpr() {
    if (current_token_.kind != tok_left_paren) {
        return LogError("Expected '(' in tensor.create");
    }
    getNextToken(); // eat (
    
    auto rows = parseExpression();
    if (!rows) {
        return nullptr;
    }
    
    if (current_token_.kind != tok_comma) {
        return LogError("Expected ',' after rows in tensor.create");
    }
    getNextToken(); // eat ,
    
    auto cols = parseExpression();
    if (!cols) {
        return nullptr;
    }
    
    if (current_token_.kind != tok_comma) {
        return LogError("Expected ',' after cols in tensor.create");
    }
    getNextToken(); // eat ,
    
    // 解析数组表达式
    auto values = parseArrayExpr();
    if (!values) {
        return nullptr;
    }
    
    if (current_token_.kind != tok_right_paren) {
        return LogError("Expected ')' after values in tensor.create");
    }
    getNextToken(); // eat )
    
    return std::make_unique<TensorCreateExprASTImpl>(
        std::move(rows),
        std::move(cols),
        std::move(values)
    );
}

std::unique_ptr<ExprAST> Parser::parseArrayExpr() {
    if (current_token_.kind != tok_left_bracket) {
        return LogError("Expected '[' to start array expression");
    }
    getNextToken(); // eat [

    std::vector<std::unique_ptr<ExprAST>> elements;
    if (current_token_.kind != tok_right_bracket) {
        while (true) {
            if (auto element = parseExpression()) {
                elements.push_back(std::move(element));
            } else {
                return nullptr;
            }

            if (current_token_.kind == tok_right_bracket) {
                break;
            }

            if (current_token_.kind != tok_comma) {
                return LogError("Expected ',' or ']' in array expression");
            }
            getNextToken(); // eat ,
        }
    }

    getNextToken(); // eat ]

    return std::make_unique<ArrayExprASTImpl>(std::move(elements));
}

std::unique_ptr<ExprAST> Parser::parseDeviceTransferExpr(std::unique_ptr<ExprAST> tensor) {
    if (current_token_.kind != tok_dot) {
        return LogError("Expected '.' before 'to'");
    }
    getNextToken(); // eat .
    
    if (current_token_.value != "to") {
        return LogError("Expected 'to' method call");
    }
    getNextToken(); // eat 'to'
    
    if (current_token_.kind != tok_left_paren) {
        return LogError("Expected '(' after 'to'");
    }
    getNextToken(); // eat (
    
    if (current_token_.kind != tok_string) {
        return LogError("Expected device name string in to()");
    }
    std::string deviceName = current_token_.value;
    getNextToken(); // eat device name
    
    if (current_token_.kind != tok_right_paren) {
        return LogError("Expected ')' after device name");
    }
    getNextToken(); // eat )
    
    return std::make_unique<DeviceTransferExprAST>(std::move(tensor), deviceName);
}

} // namespace matrix
