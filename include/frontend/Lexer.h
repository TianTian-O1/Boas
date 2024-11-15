#ifndef MATRIX_LEXER_H
#define MATRIX_LEXER_H

#include <string>
#include <vector>

namespace matrix {

enum TokenKind {
    tok_eof = -1,
    tok_def = -2,
    tok_identifier = -3,
    tok_number = -4,
    tok_import = -5,
    tok_from = -6,
    tok_tensor = -7,
    tok_matmul = -8,
    tok_equal = -9,
    tok_comma = -10,
    tok_left_bracket = -11,
    tok_right_bracket = -12,
    tok_left_paren = -13,
    tok_right_paren = -14,
    tok_colon = -15,
    tok_dedent = -16,
    tok_indent = -17,
    tok_newline = -18,
    tok_print = -19,
    tok_dot = -20,
    tok_return = -21,
    tok_linalg = -22,
    
    EOF_TOKEN = tok_eof,
    NEWLINE = tok_newline
};

struct Token {
    TokenKind kind;
    std::string value;
    
    Token(TokenKind k = tok_eof, std::string v = "") : kind(k), value(v) {}
};

class Lexer {
public:
    Lexer(std::string input) : input_(input), current_(0) {}
    Token getNextToken();
    
private:
    std::string input_;
    size_t current_;
    
    bool isAtEnd() const { return current_ >= input_.length(); }
    char peek() const { return isAtEnd() ? '\0' : input_[current_]; }
    char advance() { return input_[current_++]; }
    Token createToken(TokenKind kind, const std::string& value = "");
    std::string readIdentifier();
    std::string readNumber();
    void skipWhitespace();
};

std::string tokenKindToString(TokenKind kind);

} // namespace matrix

#endif // MATRIX_LEXER_H
