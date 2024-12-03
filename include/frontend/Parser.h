#ifndef MATRIX_PARSER_H
#define MATRIX_PARSER_H

#include "AST.h"
#include "Lexer.h"
#include <memory>
#include <string>
#include <map>
#include <vector>

namespace matrix {

class Parser {
private:
    Lexer lexer;
    Token current_token_;
    std::map<std::string, int> binop_precedence_;
    
    void getNextToken() { current_token_ = lexer.getNextToken(); }
    int getTokPrecedence();
    std::unique_ptr<ExprAST> LogError(const std::string &str);
    
public:
    Parser(const std::string &source) : lexer(source) {
        initializePrecedence();
        getNextToken();
    }
    
    Parser(Lexer &lex) : lexer(lex) {
        initializePrecedence();
        getNextToken();
    }

    void initializePrecedence() {
        binop_precedence_["+"] = 20;
        binop_precedence_["-"] = 20;
        binop_precedence_["*"] = 40;
        binop_precedence_["/"] = 40;
    }
    
    std::vector<std::unique_ptr<ExprAST>> parse() {
        std::vector<std::unique_ptr<ExprAST>> program;
        while (current_token_.kind != tok_eof) {
            if (auto expr = parseExpression()) {
                program.push_back(std::move(expr));
            }
            // Skip any newlines between expressions
            while (current_token_.kind == tok_newline) {
                getNextToken();
            }
        }
        return program;
    }
    
    std::unique_ptr<ExprAST> parseExpression();
    std::unique_ptr<ExprAST> parsePrimary();
    std::unique_ptr<ExprAST> parseParenExpr();
    std::unique_ptr<ExprAST> parseIdentifierExpr();
    std::unique_ptr<ExprAST> parseNumberExpr();
    std::unique_ptr<ExprAST> parseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS);
    std::unique_ptr<ExprAST> parseTensorExpr();
    std::unique_ptr<ExprAST> parseTensorRandomExpr();
    std::unique_ptr<ExprAST> parseTensorCreateExpr();
    std::unique_ptr<ExprAST> parseMatmulExpr();
    std::unique_ptr<ExprAST> parseTimeCallExpr();
    std::unique_ptr<ExprAST> parseForExpr();
    std::unique_ptr<ExprAST> parseWhileExpr();
    std::unique_ptr<ExprAST> parseBreakExpr();
    std::unique_ptr<ExprAST> parseContinueExpr();
    std::unique_ptr<ExprAST> parseListCompExpr();
    std::unique_ptr<ExprAST> parseCompForExpr();
    std::unique_ptr<ExprAST> parseCompIfExpr();
    std::unique_ptr<ExprAST> parseReturnExpr();
    std::unique_ptr<ExprAST> parseModuleExpr();
};

} // namespace matrix

#endif // MATRIX_PARSER_H
