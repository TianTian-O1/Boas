#ifndef BOAS_PYTHON_AST_BUILDER_H
#define BOAS_PYTHON_AST_BUILDER_H

#include <memory>
#include <string>
#include <Python.h>
#include "PythonASTNodes.h"

namespace boas {
namespace python {

class PythonASTBuilder {
private:
    bool initialized;
    PyObject* astModule;

    std::unique_ptr<ModuleNode> buildModule(PyObject* node);
    std::unique_ptr<FunctionDefNode> buildFunctionDef(PyObject* node);
    std::unique_ptr<ExprNode> buildExpr(PyObject* node);
    std::unique_ptr<BinOpNode> buildBinOp(PyObject* node);
    std::unique_ptr<NumNode> buildNum(PyObject* node);
    std::unique_ptr<NameNode> buildName(PyObject* node);
    std::unique_ptr<AssignNode> buildAssign(PyObject* node);
    std::unique_ptr<CallNode> buildCall(PyObject* node);
    std::unique_ptr<ListNode> buildList(PyObject* node);
    std::unique_ptr<ForNode> buildFor(PyObject* node);
    std::unique_ptr<ReturnNode> buildReturn(PyObject* node);
    std::unique_ptr<PythonASTNode> buildConstant(PyObject* node);

public:
    PythonASTBuilder();
    ~PythonASTBuilder();

    // 从Python源代码构建AST
    std::unique_ptr<PythonASTNode> buildFromSource(const std::string& source);
    std::unique_ptr<PythonASTNode> buildFromPyObject(PyObject* node);
};

} // namespace python
} // namespace boas

#endif // BOAS_PYTHON_AST_BUILDER_H 