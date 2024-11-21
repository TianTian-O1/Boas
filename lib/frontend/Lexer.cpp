#include "frontend/Lexer.h"
#include "Debug.h"
#include <iostream>
#include <cctype>
#include <map>

namespace matrix {

Token Lexer::createToken(TokenKind kind, const std::string& value) {
    Token token(kind, value);
    Debug::logf(DEBUG_LEXER, "Generated Token: {kind: %s, value: '%s'}", 
                tokenKindToString(kind).c_str(), value.c_str());
    return token;
}

std::string Lexer::readIdentifier() {
    std::string identifier;
    while (current_ < input_.length() && (std::isalnum(input_[current_]) || input_[current_] == '_')) {
        identifier += input_[current_++];
    }
    Debug::logf(DEBUG_LEXER, "Found identifier: %s", identifier.c_str());
    return identifier;
}

std::string Lexer::readNumber() {
    std::string number;
    while (current_ < input_.length() && std::isdigit(input_[current_])) {
        number += input_[current_++];
    }
    Debug::logf(DEBUG_LEXER, "Found number: %s", number.c_str());
    return number;
}

void Lexer::skipWhitespace() {
    while (current_ < input_.length()) {
        char c = input_[current_];
        if (c == ' ' || c == '\t' || c == '\r') {
            current_++;
        } else if (c == '#') {
            while (current_ < input_.length() && input_[current_] != '\n') {
                current_++;
            }
        } else {
            break;
        }
    }
}

Token Lexer::getNextToken() {
    skipWhitespace();
    
    if (current_ >= input_.length()) {
        return Token{tok_eof, ""};
    }
    
    char c = input_[current_];
    
    if (c == '\n') {
        current_++;
        return Token{tok_newline, "\n"};
    }
    
    if (std::isalpha(input_[current_])) {
        std::string identifier = readIdentifier();
        
        static const std::map<std::string, TokenKind> keywords = {
            {"tensor", tok_tensor},
            {"create", tok_create},
            {"random", tok_random},
            {"matmul", tok_matmul},
            {"print", tok_print},
            {"def", tok_def},
            {"import", tok_import},
            {"from", tok_from},
            {"return", tok_return},
            {"linalg", tok_linalg},
            {"time", tok_time},
            {"now", tok_now}
        };
        
        auto it = keywords.find(identifier);
        if (it != keywords.end()) {
            return createToken(it->second, identifier);
        }
        
        return createToken(tok_identifier, identifier);
    }
    
    if (std::isdigit(input_[current_])) {
        return createToken(tok_number, readNumber());
    }
    
    char currentChar = input_[current_++];
    switch (currentChar) {
        case '{': return createToken(tok_left_brace, "{");
        case '}': return createToken(tok_right_brace, "}");
        case '=': return createToken(tok_equal, "=");
        case ',': return createToken(tok_comma, ",");
        case '[': return createToken(tok_left_bracket, "[");
        case ']': return createToken(tok_right_bracket, "]");
        case '(': return createToken(tok_left_paren, "(");
        case ')': return createToken(tok_right_paren, ")");
        case ':': return createToken(tok_colon, ":");
        case '.': return createToken(tok_dot, ".");
        case '-': return createToken(tok_minus, "-");
        default: return createToken(tok_eof, std::string(1, currentChar));
    }
}

std::string tokenKindToString(TokenKind kind) {
    static const std::map<TokenKind, std::string> tokenNames = {
        {tok_eof, "EOF"},
        {tok_def, "DEF"},
        {tok_identifier, "IDENTIFIER"},
        {tok_number, "NUMBER"},
        {tok_import, "IMPORT"},
        {tok_from, "FROM"},
        {tok_tensor, "TENSOR"},
        {tok_matmul, "MATMUL"},
        {tok_equal, "EQUAL"},
        {tok_comma, "COMMA"},
        {tok_left_bracket, "LBRACKET"},
        {tok_right_bracket, "RBRACKET"},
        {tok_left_paren, "LPAREN"},
        {tok_right_paren, "RPAREN"},
        {tok_colon, "COLON"},
        {tok_dedent, "DEDENT"},
        {tok_indent, "INDENT"},
        {tok_newline, "NEWLINE"},
        {tok_print, "PRINT"},
        {tok_dot, "DOT"},
        {tok_return, "RETURN"},
        {tok_linalg, "LINALG"},
        {tok_create, "CREATE"},
        {tok_random, "RANDOM"},
        {tok_left_brace, "LBRACE"},
        {tok_right_brace, "RBRACE"},
        {tok_time, "TIME"},
        {tok_now, "NOW"},
        {tok_minus, "MINUS"}
    };
    
    auto it = tokenNames.find(kind);
    return it != tokenNames.end() ? it->second : "UNKNOWN";
}

} // namespace matrix