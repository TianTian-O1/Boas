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
    while (pos_ < input_.length() && (std::isalnum(input_[pos_]) || input_[pos_] == '_')) {
        identifier += input_[pos_++];
    }
    Debug::logf(DEBUG_LEXER, "Found identifier: %s", identifier.c_str());
    return identifier;
}

std::string Lexer::readNumber() {
    std::string number;
    while (pos_ < input_.length() && std::isdigit(input_[pos_])) {
        number += input_[pos_++];
    }
    Debug::logf(DEBUG_LEXER, "Found number: %s", number.c_str());
    return number;
}

void Lexer::skipWhitespace() {
    while (pos_ < input_.length() && std::isspace(input_[pos_])) {
        if (input_[pos_] == '\n') {
            pos_++;
            return;
        }
        pos_++;
    }
}

Token Lexer::getNextToken() {
    // Skip whitespace
    while (pos_ < input_.length() && std::isspace(input_[pos_])) {
        if (input_[pos_] == '\n') {
            pos_++;
            return createToken(tok_newline, "\n");
        }
        pos_++;
    }
    
    // Check for EOF
    if (pos_ >= input_.length()) {
        return createToken(tok_eof);
    }
    
    Debug::logf(DEBUG_LEXER, "Current char: '%c' at position %zu", input_[pos_], pos_);
    
    // Handle identifiers and keywords
    if (std::isalpha(input_[pos_])) {
        std::string identifier = readIdentifier();
        
        // Check for keywords
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
    
    // Handle numbers
    if (std::isdigit(input_[pos_])) {
        return createToken(tok_number, readNumber());
    }
    
    // Handle single-character tokens
    char currentChar = input_[pos_++];
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
    
    // Unknown character
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
