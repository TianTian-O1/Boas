#include "frontend/Parser.h"
#include "Debug.h"
#include <iostream>

namespace matrix {

std::vector<std::unique_ptr<ExprAST>> Parser::parse() {
    std::vector<std::unique_ptr<ExprAST>> program;
    
    while (current_token_.kind != tok_eof) {
        Debug::log(DEBUG_PARSER, ("Current token: " + std::to_string(current_token_.kind) + 
                  ", Value: " + current_token_.value).c_str());
        
        if (current_token_.kind == tok_newline) {
            getNextToken();  // Skip newlines
            continue;
        }
        
        std::unique_ptr<ExprAST> expr;
        switch (current_token_.kind) {
            case tok_import:
                expr = parseImport();
                break;
            case tok_from:
                expr = parseFromImport();
                break;
            case tok_def:
                expr = parseFunction();
                break;
            default:
                expr = parseExpression();
                break;
        }
        
        if (!expr) {
            Debug::log(DEBUG_PARSER, "Failed to parse expression");
            return program;
        }
        
        program.push_back(std::move(expr));
    }
    
    return program;
}

std::unique_ptr<ExprAST> Parser::parseTopLevelExpr() {
    switch (current_token_.kind) {
        case tok_import:
            return parseImport();
        case tok_from:
            return parseFromImport();
        case tok_def:
            return parseFunction();
        default:
            return parseExpression();
    }
}

std::unique_ptr<ExprAST> Parser::parseArrayLiteral() {
    Debug::log(DEBUG_PARSER, ("=== Starting Array Literal Parsing ===\n"
               "Initial token: " + tokenToString(current_token_)).c_str());
    getNextToken();  // eat '['
    
    Debug::log(DEBUG_PARSER, ("After '[': " + tokenToString(current_token_)).c_str());
    std::vector<std::unique_ptr<ExprAST>> elements;
    int elementCount = 0;
    
    while (current_token_.kind != tok_right_bracket) {
        // Skip whitespace and newlines
        while (current_token_.kind == tok_newline) {
            getNextToken();
        }
        
        Debug::log(DEBUG_PARSER, ("--- Parsing Array Element " + 
                   std::to_string(elementCount) + " ---").c_str());
        
        if (auto element = parseExpression()) {
            Debug::log(DEBUG_PARSER, "Successfully parsed array element");
            elements.push_back(std::move(element));
            elementCount++;
        } else {
            Debug::log(DEBUG_PARSER, ("Failed to parse array element " + 
                       std::to_string(elementCount) + "\nCurrent state: " + 
                       tokenToString(current_token_)).c_str());
            return nullptr;
        }
        
        Debug::log(DEBUG_PARSER, ("After element: " + tokenToString(current_token_)).c_str());
        
        // Skip whitespace and newlines before checking for comma or bracket
        while (current_token_.kind == tok_newline) {
            getNextToken();
        }
        
        if (current_token_.kind == tok_comma) {
            Debug::log(DEBUG_PARSER, "Found comma, continuing to next element");
            getNextToken();  // eat ','
            
            // Skip whitespace and newlines after comma
            while (current_token_.kind == tok_newline) {
                getNextToken();
            }
            
            if (current_token_.kind == tok_right_bracket) {
                Debug::log(DEBUG_PARSER, "Error: Trailing comma in array");
                return LogError("Unexpected ',' at end of array");
            }
        } else if (current_token_.kind != tok_right_bracket) {
            Debug::log(DEBUG_PARSER, ("Error: Expected ',' or ']', found: " + 
                       tokenToString(current_token_)).c_str());
            return LogError("Expected ',' or ']' in array literal");
        }
    }
    
    Debug::log(DEBUG_PARSER, ("=== Completed Array with " + 
               std::to_string(elementCount) + " elements ===").c_str());
    getNextToken();  // eat ']'
    
    return std::make_unique<ArrayExprAST>(std::move(elements));
}

std::unique_ptr<ExprAST> Parser::parseTensorExpr() {
    Debug::log(DEBUG_PARSER, "Parsing tensor expression");
    
    auto tensorExpr = std::make_unique<VariableExprAST>("tensor");
    getNextToken(); // eat 'tensor'
    
    if (current_token_.kind == tok_dot) {
        getNextToken(); // eat '.'
        
        if (current_token_.kind == tok_matmul) {
            getNextToken(); // eat 'matmul'
            
            if (current_token_.kind != tok_left_paren) {
                return LogError("Expected '(' after 'matmul'");
            }
            getNextToken(); // eat '('
            
            auto lhs = parsePrimary();
            if (!lhs) return nullptr;
            
            if (current_token_.kind != tok_comma) {
                return LogError("Expected ',' after first matmul argument");
            }
            getNextToken(); // eat ','
            
            auto rhs = parsePrimary();
            if (!rhs) return nullptr;
            
            if (current_token_.kind != tok_right_paren) {
                return LogError("Expected ')' after matmul arguments");
            }
            getNextToken(); // eat ')'
            
            return std::make_unique<MatmulExprAST>(std::move(lhs), std::move(rhs));
        }
        else if (current_token_.kind == tok_create) {
            Debug::log(DEBUG_PARSER, "Parsing tensor.create expression");
            getNextToken(); // eat 'create'
            
            if (current_token_.kind != tok_left_paren) {
                return LogError("Expected '(' after 'create'");
            }
            getNextToken(); // eat '('
            
            // Parse dimensions
            auto rows = parseNumber();
            if (!rows) return nullptr;
            
            if (current_token_.kind != tok_comma) {
                return LogError("Expected ',' after rows dimension");
            }
            getNextToken(); // eat ','
            
            auto cols = parseNumber();
            if (!cols) return nullptr;
            
            if (current_token_.kind != tok_right_paren) {
                return LogError("Expected ')' after dimensions");
            }
            getNextToken(); // eat ')'
            
            if (current_token_.kind != tok_left_brace) {
                return LogError("Expected '{' after dimensions");
            }
            getNextToken(); // eat '{'
            
            // Parse values
            std::vector<std::unique_ptr<ExprAST>> values;
            while (current_token_.kind != tok_right_brace) {
                if (auto val = parseNumber()) {
                    values.push_back(std::move(val));
                    
                    if (current_token_.kind == tok_comma) {
                        getNextToken(); // eat ','
                    } else if (current_token_.kind != tok_right_brace) {
                        return LogError("Expected ',' or '}' after value");
                    }
                } else {
                    return nullptr;
                }
            }
            getNextToken(); // eat '}'
            
            return std::make_unique<TensorCreateExprAST>(
                std::move(rows),
                std::move(cols),
                std::move(values)
            );
        }
        
        return LogError("Expected 'create' or 'matmul' after 'tensor.'");
    }
    
    return LogError("Expected '.' after 'tensor'");
}

std::unique_ptr<ExprAST> Parser::parseMatmulExpr() {
    Debug::log(DEBUG_PARSER, "Parsing matmul expression");
    getNextToken(); // eat 'matmul'
    
    if (current_token_.kind != tok_left_paren) {
        return LogError("Expected '(' after 'matmul'");
    }
    getNextToken(); // eat '('
    
    auto lhs = parsePrimary();
    if (!lhs) return nullptr;
    
    if (current_token_.kind != tok_comma) {
        return LogError("Expected ',' after first matmul argument");
    }
    getNextToken(); // eat ','
    
    auto rhs = parsePrimary();
    if (!rhs) return nullptr;
    
    if (current_token_.kind != tok_right_paren) {
        return LogError("Expected ')' after matmul arguments");
    }
    getNextToken(); // eat ')'
    
    auto expr = std::make_unique<MatmulExprAST>(std::move(lhs), std::move(rhs));
    expr->getLHS()->setParent(expr.get());
    expr->getRHS()->setParent(expr.get());
    return expr;
}

std::unique_ptr<ExprAST> Parser::parseIdentifierExpr() {
    std::string name = current_token_.value;
    getNextToken(); // eat identifier
    
    // 检查是否是赋值语句
    if (current_token_.kind == tok_equal) {
        getNextToken(); // eat '='
        auto value = parseExpression();
        if (!value) return nullptr;
        return std::make_unique<AssignmentExprAST>(name, std::move(value));
    }
    
    return std::make_unique<VariableExprAST>(name);
}

std::vector<std::unique_ptr<ExprAST>> Parser::parseCallArgs() {
    std::vector<std::unique_ptr<ExprAST>> args;
    
    if (current_token_.kind != tok_right_paren) {
        while (1) {
            if (auto arg = parsePrimary()) {
                args.push_back(std::move(arg));
            } else {
                return std::vector<std::unique_ptr<ExprAST>>();
            }
            
            if (current_token_.kind == tok_right_paren) {
                break;
            }
            
            if (current_token_.kind != tok_comma) {
                LogError("Expected ',' or ')' in argument list");
                return std::vector<std::unique_ptr<ExprAST>>();
            }
            getNextToken();
        }
    }
    return args;
}

std::unique_ptr<ExprAST> Parser::LogError(const std::string &str) {
    Debug::log(DEBUG_PARSER, str.c_str());
    return nullptr;
}

std::unique_ptr<ExprAST> Parser::parseTensor() {
    // 跳过 tensor 关键字
    getNextToken();  // eat 'tensor'
    
    std::vector<std::unique_ptr<ExprAST>> elements;
    
    // 解析数组内容
    if (current_token_.kind != tok_left_bracket) {
        return LogError("Expected '[' in tensor expression");
    }
    getNextToken();  // eat '['
    
    while (current_token_.kind != tok_right_bracket) {
        if (auto element = parseExpression()) {
            elements.push_back(std::move(element));
        } else {
            return nullptr;
        }
        
        if (current_token_.kind == tok_comma) {
            getNextToken();  // eat ','
        } else if (current_token_.kind != tok_right_bracket) {
            return LogError("Expected ',' or ']' in tensor expression");
        }
    }
    getNextToken();  // eat ']'
    
    return std::make_unique<TensorExprAST>(std::move(elements));
}

std::unique_ptr<ImportAST> Parser::parseImport() {
    Debug::log(DEBUG_PARSER, "Parsing import statement");
    getNextToken();  // eat 'import'
    
    // Add debug information about the current token
    Debug::log(DEBUG_PARSER, ("After import token: kind=" + std::to_string(current_token_.kind) + 
              ", value='" + current_token_.value + "'").c_str());
              
    // Allow both identifiers and certain keywords as module names
    if (current_token_.kind != tok_identifier && 
        current_token_.kind != tok_tensor &&  // Allow 'tensor' as module name
        current_token_.kind != tok_matmul) {  // Allow 'matmul' as module name
        Debug::log(DEBUG_PARSER, "Expected module name after import");
        return nullptr;
    }
    
    std::string moduleName = current_token_.value;
    Debug::log(DEBUG_PARSER, ("Found module name: " + moduleName).c_str());
    getNextToken();  // eat module name
    
    // 处理换行符或文件结束
    if (current_token_.kind != tok_newline && current_token_.kind != tok_eof) {
        Debug::log(DEBUG_PARSER, ("Unexpected token after module name: kind=" + 
                  std::to_string(current_token_.kind) + ", value='" + 
                  current_token_.value + "'").c_str());
        return nullptr;
    }
    
    if (current_token_.kind == tok_newline) {
        getNextToken();  // eat newline
    }
    
    return std::make_unique<ImportAST>(moduleName);
}

std::unique_ptr<ImportAST> Parser::parseFromImport() {
    Debug::log(DEBUG_PARSER, "Parsing from import statement");
    getNextToken();  // eat 'from'
    
    // Allow both identifiers and certain keywords as module names
    if (current_token_.kind != tok_identifier && 
        current_token_.kind != tok_tensor &&  // Allow 'tensor' as module name
        current_token_.kind != tok_matmul) {  // Allow 'matmul' as module name
        Debug::log(DEBUG_PARSER, "Expected module name after from");
        return nullptr;
    }
    
    std::string moduleName = current_token_.value;
    Debug::log(DEBUG_PARSER, ("Found module name in from statement: " + moduleName).c_str());
    getNextToken();  // eat module name
    
    if (current_token_.kind != tok_import) {
        Debug::log(DEBUG_PARSER, "Expected 'import' after module name in from statement");
        return nullptr;
    }
    getNextToken();  // eat 'import'
    
    if (current_token_.kind != tok_identifier && 
        current_token_.kind != tok_matmul) {  // Allow 'matmul' as function name
        Debug::log(DEBUG_PARSER, "Expected function name after import");
        return nullptr;
    }
    
    std::string funcName = current_token_.value;
    Debug::log(DEBUG_PARSER, ("Found function name: " + funcName).c_str());
    getNextToken();  // eat function name
    
    // 处理换行符或文件结束
    if (current_token_.kind != tok_newline && current_token_.kind != tok_eof) {
        Debug::log(DEBUG_PARSER, "Expected newline after from import statement");
        return nullptr;
    }
    getNextToken();  // eat newline
    
    return std::make_unique<ImportAST>(moduleName, funcName);
}

std::unique_ptr<FunctionAST> Parser::parseFunction() {
    Debug::log(DEBUG_PARSER, "Parsing function definition");
    getNextToken(); // eat 'def'
    
    if (current_token_.kind != tok_identifier) {
        Debug::log(DEBUG_PARSER, "Expected function name after 'def'");
        return nullptr;
    }
    std::string func_name = current_token_.value;
    getNextToken(); // eat function name
    
    if (current_token_.kind != tok_left_paren) {
        Debug::log(DEBUG_PARSER, "Expected '(' after function name");
        return nullptr;
    }
    getNextToken(); // eat '('
    
    std::vector<std::string> args;
    while (current_token_.kind != tok_right_paren) {
        if (current_token_.kind != tok_identifier) {
            Debug::log(DEBUG_PARSER, "Expected parameter name");
            return nullptr;
        }
        args.push_back(current_token_.value);
        getNextToken(); // eat parameter
        
        if (current_token_.kind == tok_comma) {
            getNextToken(); // eat ','
        }
    }
    getNextToken(); // eat ')'
    
    if (current_token_.kind != tok_colon) {
        Debug::log(DEBUG_PARSER, "Expected ':' after function parameters");
        return nullptr;
    }
    getNextToken(); // eat ':'
    
    // Handle newline after colon
    if (current_token_.kind != tok_newline) {
        Debug::log(DEBUG_PARSER, "Expected newline after ':'");
        return nullptr;
    }
    getNextToken(); // eat newline
    
    std::vector<std::unique_ptr<ExprAST>> body;
    while (current_token_.kind != tok_eof && current_token_.kind != tok_def) {
        // Skip empty lines
        if (current_token_.kind == tok_newline) {
            getNextToken();
            continue;
        }
        
        if (auto expr = parseExpression()) {
            body.push_back(std::move(expr));
            
            // Expect newline after each expression
            if (current_token_.kind != tok_newline && current_token_.kind != tok_eof) {
                Debug::log(DEBUG_PARSER, "Expected newline after expression in function body");
                return nullptr;
            }
            if (current_token_.kind == tok_newline) {
                getNextToken(); // eat newline
            }
        } else {
            Debug::log(DEBUG_PARSER, "Failed to parse expression in function body");
            return nullptr;
        }
    }
    
    return std::make_unique<FunctionAST>(func_name, std::move(args), std::move(body));
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
    Debug::log(DEBUG_PARSER, "Parsing primary expression");
    switch (current_token_.kind) {
        case tok_identifier:
            return parseIdentifierExpr();
        case tok_number:
            return parseNumber();
        case tok_left_paren:
            return parseParenExpr();
        case tok_tensor:
            return parseTensorExpr();
        case tok_matmul:
            return parseMatmulExpr();
        case tok_print:
            return parsePrintExpr();
        default:
            Debug::log(DEBUG_PARSER, ("Unknown token when expecting an expression: kind=" + 
                      std::to_string(current_token_.kind) + ", value='" + 
                      current_token_.value + "'").c_str());
            return nullptr;
    }
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
    Debug::log(DEBUG_PARSER, "Parsing expression");
    auto lhs = parsePrimary();
    if (!lhs) return nullptr;
    
    return parseExpressionRHS(std::move(lhs));
}

std::unique_ptr<ExprAST> Parser::parseIdentifier() {
    std::string name = current_token_.value;
    getNextToken(); // eat identifier
    return std::make_unique<VariableExprAST>(name);
}

std::unique_ptr<ExprAST> Parser::parseExpressionRHS(std::unique_ptr<ExprAST> lhs) {
    // 暂时只返回 lhs，后续可以添加运算符优先级处理
    return std::move(lhs);
}

std::unique_ptr<ExprAST> Parser::parseNumber() {
    auto result = std::make_unique<NumberExprAST>(std::stod(current_token_.value));
    getNextToken(); // eat number
    return result;
}

std::unique_ptr<ExprAST> Parser::parseParenExpr() {
    getNextToken(); // eat '('
    
    auto expr = parseExpression();
    if (!expr) return nullptr;
    
    if (current_token_.kind != tok_right_paren) {
        return LogError("Expected ')'");
    }
    getNextToken(); // eat ')'
    
    return expr;
}

std::unique_ptr<ExprAST> Parser::parsePrintExpr() {
    Debug::log(DEBUG_PARSER, "Parsing print expression");
    getNextToken();  // eat 'print'
    
    if (current_token_.kind != tok_left_paren) {
        return LogError("Expected '(' after 'print'");
    }
    getNextToken();  // eat '('
    
    Debug::log(DEBUG_PARSER, "Parsing print argument");
    auto arg = parseExpression();
    if (!arg) {
        return LogError("Failed to parse print argument");
    }
    
    if (current_token_.kind != tok_right_paren) {
        return LogError("Expected ')' after print argument");
    }
    getNextToken();  // eat ')'
    
    Debug::log(DEBUG_PARSER, "Successfully parsed print expression");
    return std::make_unique<PrintExprAST>(std::move(arg));
}

// Helper function to convert token to string representation
std::string Parser::tokenToString(const Token& token) const {
    return "Token{kind=" + std::to_string(token.kind) + 
           ", value='" + token.value + "'}";
}

} // namespace matrix
