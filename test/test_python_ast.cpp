#include "frontend/PythonAST.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace matrix;

std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void printASTNode(PyObject* node, int indent = 0) {
    PyObject* astModule = PyImport_ImportModule("ast");
    if (!astModule) return;
    
    // 获取节点类型
    PyObject* nodeType = PyObject_GetAttrString(node, "__class__");
    PyObject* typeName = PyObject_GetAttrString(nodeType, "__name__");
    std::string nodeTypeName = PyUnicode_AsUTF8(typeName);
    
    // 打印缩进和节点类型
    std::string indentStr(indent * 2, ' ');
    std::cout << indentStr << nodeTypeName << ":\n";
    
    // 获取节点的所有属性
    PyObject* attrs = PyObject_Dir(node);
    Py_ssize_t size = PyList_Size(attrs);
    
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* attr = PyList_GetItem(attrs, i);
        std::string attrName = PyUnicode_AsUTF8(attr);
        
        // 跳过私有属性和内部属性
        if (attrName[0] == '_') continue;
        
        PyObject* value = PyObject_GetAttrString(node, attrName.c_str());
        if (!value) continue;
        
        // 检查是否是AST节点或节点列表
        bool isASTNode = PyObject_IsInstance(value, PyObject_GetAttrString(astModule, "AST"));
        bool isList = PyList_Check(value);
        
        std::cout << indentStr << "  " << attrName << ": ";
        
        if (isASTNode) {
            std::cout << "\n";
            printASTNode(value, indent + 2);
        } else if (isList) {
            std::cout << "\n";
            Py_ssize_t listSize = PyList_Size(value);
            for (Py_ssize_t j = 0; j < listSize; j++) {
                PyObject* item = PyList_GetItem(value, j);
                if (PyObject_IsInstance(item, PyObject_GetAttrString(astModule, "AST"))) {
                    printASTNode(item, indent + 2);
                }
            }
        } else {
            PyObject* str = PyObject_Str(value);
            if (str) {
                std::cout << PyUnicode_AsUTF8(str) << "\n";
                Py_DECREF(str);
            }
        }
        
        Py_DECREF(value);
    }
    
    Py_DECREF(nodeType);
    Py_DECREF(typeName);
    Py_DECREF(attrs);
    Py_DECREF(astModule);
}

int main() {
    PythonAST ast;
    
    // 初始化AST系统
    if (!ast.initialize()) {
        std::cerr << "Failed to initialize Python AST system\n";
        return 1;
    }
    
    try {
        // 读取并解析test_func.txt
        std::string code = readFile("test/test_func.txt");
        PyObject* tree = ast.parseString(code);
        
        if (tree) {
            std::cout << "Original code:\n" << code << "\n\n";
            std::cout << "Generated AST:\n" << ast.toSource(tree) << std::endl;
            
            std::cout << "\nDetailed AST Structure:\n";
            printASTNode(tree);
            
            Py_DECREF(tree);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 