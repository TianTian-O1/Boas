#include "frontend/PythonASTParser.h"
#include <iostream>
#include <string>

void testTensorFunc() {
    std::string testCode = R"(
import tensor

def benchmark(size):
    # 4 x 4
    A4 = tensor.random(size, size)
    B4 = tensor.random(size, size)
    C4 = tensor.matmul(A4, B4)
    return C4

def main():
    sizes = [2,3,4]
    for size in sizes:
        C4 = benchmark(size)
        print(C4)
    print("successfully")

    # 使用列表初始化
    A = tensor.create(2, 2, [1,2,2,3])
    B = tensor.create(2, 2, [5,1,7,8])
    C = tensor.matmul(A, B)
    print(C)
)";

    boas::PythonASTParser parser;
    
    if (!parser.initialize()) {
        std::cerr << "Failed to initialize parser" << std::endl;
        return;
    }

    auto ast = parser.parse(testCode);
    if (!ast) {
        std::cerr << "Failed to parse tensor function code" << std::endl;
        return;
    }

    std::cout << "Parsing tensor function code successful!" << std::endl;
    ast->dump();
}

int main() {
    std::cout << "Testing tensor function parsing..." << std::endl;
    testTensorFunc();
    return 0;
} 