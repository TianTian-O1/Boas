#ifndef MATRIX_PARSER_H
#define MATRIX_PARSER_H

#include "Lexer.h"
#include "frontend/AST.h"
#include <memory>
#include <vector>

namespace matrix {

class Parser {
public:
    Parser(Lexer &lexer) : lexer_(lexer) {
        getNextToken();
    }
    std::vector<std::unique_ptr<ExprAST>> parse();

private:
    Lexer &lexer_;
    Token current_token_;

    // 辅助方法
    Token getNextToken() { return current_token_ = lexer_.getNextToken(); }
    std::string tokenToString(const Token& token) const;
    
    // 解析方法
    std::unique_ptr<ExprAST> parseTopLevelExpr();
    std::unique_ptr<ExprAST> parseNumber();
    std::unique_ptr<ExprAST> parseParenExpr();
    std::unique_ptr<ExprAST> parseIdentifierExpr();
    std::unique_ptr<ExprAST> parsePrimary();
    std::unique_ptr<ExprAST> parseExpression();
    std::unique_ptr<ExprAST> parseArrayLiteral();
    std::unique_ptr<ExprAST> parseTensorExpr();
    std::unique_ptr<ExprAST> parseMatmulExpr();
    std::unique_ptr<ExprAST> parseTensor();
    
    // 错误处理
    std::unique_ptr<ExprAST> LogError(const std::string &str);

    // 语句级解析方法
    std::unique_ptr<ImportAST> parseImport();
    std::unique_ptr<ImportAST> parseFromImport();
    std::unique_ptr<FunctionAST> parseFunction();

    // 辅助解析方法
    std::vector<std::unique_ptr<ExprAST>> parseCallArgs();
    std::vector<std::string> parseFunctionArgs();

    std::unique_ptr<ExprAST> parseIdentifier();
    std::unique_ptr<ExprAST> parseExpressionRHS(std::unique_ptr<ExprAST> lhs);
    std::unique_ptr<ExprAST> parsePrintExpr();
};

} // namespace matrix

#endif // MATRIX_PARSER_H
