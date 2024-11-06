#!/bin/bash

# 创建目录结构
mkdir -p src/{frontend,ast,parser,codegen,optimizer,runtime}
mkdir -p include/boas
mkdir -p lib
mkdir -p test
mkdir -p examples
mkdir -p build

# 创建主要的头文件
cat > include/boas/Lexer.h << 'EOF'
#pragma once
#include <string>
#include <vector>

namespace boas {

enum TokenType {
  TOK_EOF = 0,
  TOK_PRINT,
  TOK_STRING,
  TOK_LPAREN,
  TOK_RPAREN
};

struct Token {
  TokenType type;
  std::string value;
};

class Lexer {
public:
  Lexer(const std::string &input) : input_(input), pos_(0) {}
  Token getNextToken();

private:
  std::string input_;
  size_t pos_;
  void skipWhitespace();
};

} // namespace boas
EOF

# 创建词法分析器实现
cat > src/frontend/Lexer.cpp << 'EOF'
#include "boas/Lexer.h"

namespace boas {

void Lexer::skipWhitespace() {
  while (pos_ < input_.length() && isspace(input_[pos_])) {
    pos_++;
  }
}

Token Lexer::getNextToken() {
  skipWhitespace();

  if (pos_ >= input_.length()) {
    return {TOK_EOF, ""};
  }

  if (input_.substr(pos_, 5) == "print") {
    pos_ += 5;
    return {TOK_PRINT, "print"};
  }

  if (input_[pos_] == '(') {
    pos_++;
    return {TOK_LPAREN, "("};
  }

  if (input_[pos_] == ')') {
    pos_++;
    return {TOK_RPAREN, ")"};
  }

  if (input_[pos_] == '"') {
    std::string str;
    pos_++; // 跳过开始的引号
    while (pos_ < input_.length() && input_[pos_] != '"') {
      str += input_[pos_++];
    }
    pos_++; // 跳过结束的引号
    return {TOK_STRING, str};
  }

  return {TOK_EOF, ""};
}

} // namespace boas
EOF

# 创建AST头文件
cat > include/boas/AST.h << 'EOF'
#pragma once
#include <string>
#include <memory>

namespace boas {

class ASTNode {
public:
  virtual ~ASTNode() = default;
};

class PrintNode : public ASTNode {
public:
  PrintNode(const std::string &value) : value_(value) {}
  const std::string &getValue() const { return value_; }

private:
  std::string value_;
};

} // namespace boas
EOF

# 创建Parser头文件
cat > include/boas/Parser.h << 'EOF'
#pragma once
#include "boas/Lexer.h"
#include "boas/AST.h"
#include <memory>

namespace boas {

class Parser {
public:
  Parser(Lexer &lexer) : lexer_(lexer) {}
  std::unique_ptr<ASTNode> parse();

private:
  Lexer &lexer_;
  Token currentToken_;
  void advance();
};

} // namespace boas
EOF

# 创建Parser实现
cat > src/parser/Parser.cpp << 'EOF'
#include "boas/Parser.h"

namespace boas {

void Parser::advance() {
  currentToken_ = lexer_.getNextToken();
}

std::unique_ptr<ASTNode> Parser::parse() {
  advance();
  
  if (currentToken_.type == TOK_PRINT) {
    advance();
    if (currentToken_.type == TOK_LPAREN) {
      advance();
      if (currentToken_.type == TOK_STRING) {
        std::string value = currentToken_.value;
        advance();
        if (currentToken_.type == TOK_RPAREN) {
          return std::make_unique<PrintNode>(value);
        }
      }
    }
  }
  
  return nullptr;
}

} // namespace boas
EOF

# 创建MLIR方言定义
cat > include/boas/BoasOps.td << 'EOF'
#ifndef BOAS_OPS
#define BOAS_OPS

include "mlir/IR/OpBase.td"

def Boas_Dialect : Dialect {
  let name = "boas";
  let summary = "A dialect for Boas language";
  let cppNamespace = "::boas";
}

def PrintOp : Op<Boas_Dialect, "print"> {
  let summary = "Print operation";
  let arguments = (ins StrAttr:$value);
  let assemblyFormat = "$value attr-dict";
}

#endif // BOAS_OPS
EOF

# 创建主CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.13.4)
project(boas LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src)
EOF

# 创建src的CMakeLists.txt
cat > src/CMakeLists.txt << 'EOF'
add_library(BoasLib
  frontend/Lexer.cpp
  parser/Parser.cpp
)

target_include_directories(BoasLib PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

add_executable(boas main.cpp)
target_link_libraries(boas PRIVATE BoasLib)
EOF

# 创建主程序入口
cat > src/main.cpp << 'EOF'
#include "boas/Lexer.h"
#include "boas/Parser.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
    return 1;
  }

  std::ifstream file(argv[1]);
  std::stringstream buffer;
  buffer << file.rdbuf();
  
  boas::Lexer lexer(buffer.str());
  boas::Parser parser(lexer);
  
  auto ast = parser.parse();
  if (auto printNode = dynamic_cast<boas::PrintNode*>(ast.get())) {
    std::cout << printNode->getValue() << std::endl;
  }

  return 0;
}
EOF

# 创建构建脚本
cat > build.sh << 'EOF'
#!/bin/bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
EOF

# 创建示例文件
cat > examples/hello.bs << 'EOF'
print("Hello, Boas!")
EOF

# 设置执行权限
chmod +x build.sh
chmod +x create-boas-project.sh

echo "项目创建完成！" 