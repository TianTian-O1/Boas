#include "frontend/Lexer.h"
#include "Debug.h"
#include <iostream>
#include <cctype>

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
        if (c == ' ' || c == '\t') {
            current_++;
        } else if (c == '#') {
            while (current_ < input_.length() && input_[current_] != '\n') {
                current_++;
            }
            break;
        } else if (c == '\n') {
            break;
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
        
        if (identifier == "def") return createToken(tok_def, identifier);
        if (identifier == "import") return createToken(tok_import, identifier);
        if (identifier == "from") return createToken(tok_from, identifier);
        if (identifier == "tensor") return createToken(tok_tensor, identifier);
        if (identifier == "matmul") return createToken(tok_matmul, identifier);
        if (identifier == "print") return createToken(tok_print, identifier);
        if (identifier == "return") return createToken(tok_return, identifier);
        if (identifier == "linalg") return createToken(tok_linalg, identifier);
        
        return createToken(tok_identifier, identifier);
    }
    
    if (std::isdigit(input_[current_])) {
        return createToken(tok_number, readNumber());
    }
    
    char currentChar = input_[current_++];
    switch (currentChar) {
        case '=': return createToken(tok_equal, "=");
        case ',': return createToken(tok_comma, ",");
        case '[': return createToken(tok_left_bracket, "[");
        case ']': return createToken(tok_right_bracket, "]");
        case '(': return createToken(tok_left_paren, "(");
        case ')': return createToken(tok_right_paren, ")");
        case ':': return createToken(tok_colon, ":");
        case '.': return createToken(tok_dot, ".");
    }
    
    std::string unknown(1, currentChar);
    return createToken(tok_eof, unknown);
}

std::string tokenKindToString(TokenKind kind) {
    switch (kind) {
        case tok_eof: return "EOF";
        case tok_def: return "DEF";
        case tok_identifier: return "IDENTIFIER";
        case tok_number: return "NUMBER";
        case tok_import: return "IMPORT";
        case tok_from: return "FROM";
        case tok_tensor: return "TENSOR";
        case tok_matmul: return "MATMUL";
        case tok_equal: return "EQUAL";
        case tok_comma: return "COMMA";
        case tok_left_bracket: return "LBRACKET";
        case tok_right_bracket: return "RBRACKET";
        case tok_left_paren: return "LPAREN";
        case tok_right_paren: return "RPAREN";
        case tok_colon: return "COLON";
        case tok_dedent: return "DEDENT";
        case tok_indent: return "INDENT";
        case tok_newline: return "NEWLINE";
        case tok_print: return "PRINT";
        case tok_dot: return "DOT";
        case tok_return: return "RETURN";
        case tok_linalg: return "LINALG";
        default: return "UNKNOWN";
    }
}

} // namespace matrix
