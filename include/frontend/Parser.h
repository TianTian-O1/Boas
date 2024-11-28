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

protected:
    // Core parsing infrastructure
    Token getNextToken() { return current_token_ = lexer_.getNextToken(); }
    std::unique_ptr<ExprAST> LogError(const std::string &str);
    std::string tokenToString(const Token& token) const;
    
    // Expression parsing methods
    std::unique_ptr<ExprAST> parseTopLevelExpr();
    std::unique_ptr<ExprAST> parsePrimary();
    std::unique_ptr<ExprAST> parseExpression();
    std::unique_ptr<ExprAST> parseExpressionRHS(std::unique_ptr<ExprAST> lhs);
    std::unique_ptr<ExprAST> parseNumber();
    std::unique_ptr<ExprAST> parseIdentifier();
    std::unique_ptr<ExprAST> parseIdentifierExpr();
    std::unique_ptr<ExprAST> parseParenExpr();
    std::unique_ptr<ExprAST> parseArrayLiteral();
    std::unique_ptr<ExprAST> parseList();
    
    // Tensor-related parsing methods
    std::unique_ptr<ExprAST> parseTensorExpr();
    std::unique_ptr<ExprAST> parseTensorCreate();
    std::unique_ptr<ExprAST> parseTensorRandom();
    std::unique_ptr<ExprAST> parseMatmulExpr();
    std::unique_ptr<ExprAST> parseRandomExpr();
    
    // Statement parsing methods
    std::unique_ptr<ImportAST> parseImport();
    std::unique_ptr<ImportAST> parseFromImport();
    std::unique_ptr<FunctionAST> parseFunction();
    std::unique_ptr<AssignmentExprAST> parseAssignment();
    std::unique_ptr<ExprAST> parsePrintExpr();
    std::unique_ptr<ExprAST> parseTimeExpr();
    
    // Helper methods
    std::vector<std::unique_ptr<ExprAST>> parseCallArgs();
    bool expectToken(TokenKind kind, const std::string& message);
    bool isEndOfStatement();
    void skipNewlines();
    
    // 添加解析return语句的方法
    std::unique_ptr<ExprAST> parseReturnExpr();
    
    Lexer &lexer_;
    Token current_token_;
};

} // namespace matrix

#endif // MATRIX_PARSER_H
