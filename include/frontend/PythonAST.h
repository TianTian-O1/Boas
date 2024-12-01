#pragma once

#include <Python.h>
#include <string>
#include <memory>
#include <vector>

namespace matrix {

class PythonAST {
public:
    PythonAST();
    ~PythonAST();

    // 初始化Python AST模块（不启动解释器）
    bool initialize();

    // 从字符串生成AST
    PyObject* parseString(const std::string& code);

    // 从文件生成AST
    PyObject* parseFile(const std::string& filename);

    // AST操作函数
    PyObject* createNum(double value);
    PyObject* createStr(const std::string& value);
    PyObject* createName(const std::string& id);
    PyObject* createBinOp(PyObject* left, const std::string& op, PyObject* right);
    PyObject* createAssign(const std::string& target, PyObject* value);
    PyObject* createFunctionDef(const std::string& name,
                              const std::vector<std::string>& args,
                              const std::vector<PyObject*>& body);
    
    // AST转换为源代码
    std::string toSource(PyObject* node);

private:
    // Python AST模块
    PyObject* astModule;
    PyObject* astBuilderModule;
    
    // 初始化必要的Python对象
    void initializeModules();
    
    // 错误处理
    void handlePythonError();
};

} // namespace matrix 