#include "boas/Lexer.h"
#include "boas/Parser.h"
#include "boas/SemanticAnalyzer.h"
#include <gtest/gtest.h>

TEST(BasicTests, PrintStatement) {
  std::string input = "print(42)\n";
  boas::Lexer lexer(input);
  boas::Parser parser(lexer);
  auto ast = parser.parse();
  
  ASSERT_TRUE(ast != nullptr);
  auto* blockNode = dynamic_cast<boas::BlockNode*>(ast.get());
  ASSERT_TRUE(blockNode != nullptr);
  ASSERT_EQ(blockNode->getStatements().size(), 1);
  
  auto* printNode = dynamic_cast<boas::PrintNode*>(blockNode->getStatements()[0].get());
  ASSERT_TRUE(printNode != nullptr);
}

TEST(BasicTests, Assignment) {
  std::string input = "x = 10\n";
  boas::Lexer lexer(input);
  boas::Parser parser(lexer);
  auto ast = parser.parse();
  
  ASSERT_TRUE(ast != nullptr);
  auto* blockNode = dynamic_cast<boas::BlockNode*>(ast.get());
  ASSERT_TRUE(blockNode != nullptr);
  ASSERT_EQ(blockNode->getStatements().size(), 1);
  
  auto* assignNode = dynamic_cast<boas::AssignmentNode*>(blockNode->getStatements()[0].get());
  ASSERT_TRUE(assignNode != nullptr);
}