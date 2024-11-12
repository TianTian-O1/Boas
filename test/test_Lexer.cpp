#include "frontend/Lexer.h"
#include <iostream>

int main() {
    std::string input = R"(
import tensor
from tensor import matmul

def main():
    A = tensor([[1, 2], [3, 4]])
    B = tensor([[5, 6], [7, 8]])
    C = matmul(A, B)
    print(C)
)";

    matrix::Lexer lexer(input);
    matrix::Token token = lexer.getNextToken();

    while (token.kind != matrix::tok_eof) {
        token = lexer.getNextToken();
    }

    return 0;
}
