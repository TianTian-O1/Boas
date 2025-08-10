#ifndef BOAS_PYTHON_AST_PARSER_H
#define BOAS_PYTHON_AST_PARSER_H

#include <Python.h>
#include <memory>
#include <string>
#include <vector>
#include "frontend/ASTImpl.h"
#include "frontend/python_parser/PythonASTNodes.h"
#include "frontend/python_parser/PythonASTBuilder.h"
#include "frontend/python_parser/PythonASTVisitor.h"

namespace boas {

class PythonASTParser {
public:
    PythonASTParser();
    ~PythonASTParser();

    // 初始化和清理
    bool initialize();
    void finalize();

    // 解析Python代码
    std::unique_ptr<matrix::ExprAST> parse(const std::string& code);

private:
    // Python AST构建器
    std::unique_ptr<python::PythonASTBuilder> builder;
    
    // Boas AST转换器
    std::unique_ptr<python::BoasASTConverter> converter;

    // Python解释器状态
    bool initialized;
    PyObject* astModule;
    PyObject* builtinsModule;

    // 私有辅助方法
    void cleanup();
    std::unique_ptr<python::PythonASTNode> parseSource(const std::string& source);
    std::string getNodeType(PyObject* node);
    std::unique_ptr<matrix::ExprAST> convertPythonAST(PyObject* pythonAst);
    std::unique_ptr<matrix::ExprAST> parseNode(PyObject* node);
    
    // 节点处理方法
    std::unique_ptr<matrix::ExprAST> handleModule(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleFunctionDef(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleReturn(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleName(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleNum(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleConstant(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleBinOp(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleCall(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleAssign(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleFor(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleWhile(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleListComp(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleComprehension(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleList(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleSubscript(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleAttribute(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleImport(PyObject* node);
    std::unique_ptr<matrix::ExprAST> handleExpr(PyObject* node);
};

} // namespace boas

#endif // BOAS_PYTHON_AST_PARSER_H 