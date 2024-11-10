#include "boas/frontend/python_frontend.h"
#include "boas/ast/list_ast.h"
#include "boas/ast/number_ast.h"
#include "boas/ast/print_ast.h"
#include "boas/ast/block_ast.h"
#include <Python.h>
#include <iostream>

namespace boas {

class PythonFrontendImpl {
public:
    PythonFrontendImpl() {
        Py_Initialize();
    }
    
    ~PythonFrontendImpl() {
        Py_Finalize();
    }

    std::unique_ptr<AST> parse(const std::string& source) {
        // 创建一个 Python 模块来执行源代码
        PyObject* mainModule = PyImport_AddModule("__main__");
        PyObject* globals = PyModule_GetDict(mainModule);
        
        // 执行源代码
        PyObject* result = PyRun_String(source.c_str(), Py_file_input, globals, globals);
        if (!result) {
            PyErr_Print();
            return nullptr;
        }
        Py_DECREF(result);
        
        // 创建一个语句vector
        std::vector<std::unique_ptr<AST>> statements;
        
        // 解析 Python AST
        PyObject* ast = PyImport_ImportModule("ast");
        if (!ast) {
            PyErr_Print();
            return nullptr;
        }
        
        // 解析源代码为 Python AST
        PyObject* astTree = PyObject_CallMethod(ast, "parse", "s", source.c_str());
        if (!astTree) {
            PyErr_Print();
            Py_DECREF(ast);
            return nullptr;
        }
        
        // 遍历 AST 节点并生成对应的 Boas AST
        PyObject* body = PyObject_GetAttrString(astTree, "body");
        Py_ssize_t size = PyList_Size(body);
        
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* node = PyList_GetItem(body, i);
            // TODO: 根据节点类型生成相应的 AST 节点
            // 例如: Assign, Call, List 等
        }
        
        Py_DECREF(body);
        Py_DECREF(astTree);
        Py_DECREF(ast);
        
        // 创建 BlockAST 并传入语句vector
        return std::make_unique<BlockAST>(std::move(statements));
    }
};

PythonFrontend::PythonFrontend() : impl_(std::make_unique<PythonFrontendImpl>()) {}
PythonFrontend::~PythonFrontend() = default;

std::unique_ptr<AST> PythonFrontend::parse(const std::string& source) {
    return impl_->parse(source);
}

} // namespace boas