#include "frontend/PythonAST.h"
#include <stdexcept>

namespace matrix {

PythonAST::PythonAST() : astModule(nullptr), astBuilderModule(nullptr) {}

PythonAST::~PythonAST() {
    Py_XDECREF(astModule);
    Py_XDECREF(astBuilderModule);
}

bool PythonAST::initialize() {
    // 初始化Python（不启动完整解释器）
    Py_NoSiteFlag = 1;
    Py_InitializeEx(0);  // 0表示不初始化信号处理
    
    if (!Py_IsInitialized()) {
        return false;
    }
    
    try {
        initializeModules();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void PythonAST::initializeModules() {
    // 导入ast模块
    astModule = PyImport_ImportModule("ast");
    if (!astModule) {
        handlePythonError();
        throw std::runtime_error("Failed to import ast module");
    }
    
    // 导入ast builder模块（可选）
    astBuilderModule = PyImport_ImportModule("ast_builder");
    if (!astBuilderModule) {
        PyErr_Clear();  // 清除错误，因为这是可选的
    }
}

PyObject* PythonAST::parseString(const std::string& code) {
    PyObject* codeObj = PyUnicode_FromString(code.c_str());
    if (!codeObj) {
        handlePythonError();
        return nullptr;
    }
    
    PyObject* result = PyObject_CallMethod(astModule, "parse", "O", codeObj);
    Py_DECREF(codeObj);
    
    if (!result) {
        handlePythonError();
        return nullptr;
    }
    
    return result;
}

PyObject* PythonAST::createNum(double value) {
    PyObject* numValue = PyFloat_FromDouble(value);
    if (!numValue) {
        handlePythonError();
        return nullptr;
    }
    
    PyObject* numClass = PyObject_GetAttrString(astModule, "Num");
    if (!numClass) {
        Py_DECREF(numValue);
        handlePythonError();
        return nullptr;
    }
    
    PyObject* args = PyTuple_Pack(1, numValue);
    PyObject* result = PyObject_CallObject(numClass, args);
    
    Py_DECREF(numValue);
    Py_DECREF(numClass);
    Py_DECREF(args);
    
    if (!result) {
        handlePythonError();
        return nullptr;
    }
    
    return result;
}

PyObject* PythonAST::createName(const std::string& id) {
    PyObject* nameStr = PyUnicode_FromString(id.c_str());
    if (!nameStr) {
        handlePythonError();
        return nullptr;
    }
    
    PyObject* nameClass = PyObject_GetAttrString(astModule, "Name");
    if (!nameClass) {
        Py_DECREF(nameStr);
        handlePythonError();
        return nullptr;
    }
    
    // 创建Load上下文
    PyObject* loadClass = PyObject_GetAttrString(astModule, "Load");
    PyObject* loadCtx = PyObject_CallObject(loadClass, NULL);
    
    PyObject* args = PyTuple_Pack(2, nameStr, loadCtx);
    PyObject* result = PyObject_CallObject(nameClass, args);
    
    Py_DECREF(nameStr);
    Py_DECREF(nameClass);
    Py_DECREF(loadClass);
    Py_DECREF(loadCtx);
    Py_DECREF(args);
    
    if (!result) {
        handlePythonError();
        return nullptr;
    }
    
    return result;
}

PyObject* PythonAST::createBinOp(PyObject* left, const std::string& op, PyObject* right) {
    // 获取操作符类
    std::string opClassName = op == "+" ? "Add" :
                            op == "-" ? "Sub" :
                            op == "*" ? "Mult" :
                            op == "/" ? "Div" : "Add";
    
    PyObject* opClass = PyObject_GetAttrString(astModule, opClassName.c_str());
    if (!opClass) {
        handlePythonError();
        return nullptr;
    }
    
    PyObject* opInstance = PyObject_CallObject(opClass, NULL);
    Py_DECREF(opClass);
    
    if (!opInstance) {
        handlePythonError();
        return nullptr;
    }
    
    // 创建BinOp节点
    PyObject* binOpClass = PyObject_GetAttrString(astModule, "BinOp");
    if (!binOpClass) {
        Py_DECREF(opInstance);
        handlePythonError();
        return nullptr;
    }
    
    PyObject* args = PyTuple_Pack(3, left, opInstance, right);
    PyObject* result = PyObject_CallObject(binOpClass, args);
    
    Py_DECREF(opInstance);
    Py_DECREF(binOpClass);
    Py_DECREF(args);
    
    if (!result) {
        handlePythonError();
        return nullptr;
    }
    
    return result;
}

std::string PythonAST::toSource(PyObject* node) {
    // 使用ast.unparse替代astor
    PyObject* result = PyObject_CallMethod(astModule, "unparse", "O", node);
    if (!result) {
        handlePythonError();
        return "";
    }
    
    const char* cStr = PyUnicode_AsUTF8(result);
    std::string sourceCode = cStr ? cStr : "";
    
    Py_DECREF(result);
    return sourceCode;
}

void PythonAST::handlePythonError() {
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    
    if (value) {
        PyObject* str = PyObject_Str(value);
        if (str) {
            const char* errorMsg = PyUnicode_AsUTF8(str);
            if (errorMsg) {
                throw std::runtime_error(errorMsg);
            }
            Py_DECREF(str);
        }
    }
    
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
}

PyObject* PythonAST::createAssign(const std::string& target, PyObject* value) {
    // 创建目标名称节点
    PyObject* targetName = createName(target);
    if (!targetName) {
        return nullptr;
    }
    
    // 创建Store上下文
    PyObject* storeClass = PyObject_GetAttrString(astModule, "Store");
    if (!storeClass) {
        Py_DECREF(targetName);
        handlePythonError();
        return nullptr;
    }
    
    PyObject* storeCtx = PyObject_CallObject(storeClass, NULL);
    Py_DECREF(storeClass);
    
    if (!storeCtx) {
        Py_DECREF(targetName);
        handlePythonError();
        return nullptr;
    }
    
    // 设置目标的ctx为Store
    PyObject_SetAttrString(targetName, "ctx", storeCtx);
    Py_DECREF(storeCtx);
    
    // 创建赋值节点
    PyObject* assignClass = PyObject_GetAttrString(astModule, "Assign");
    if (!assignClass) {
        Py_DECREF(targetName);
        handlePythonError();
        return nullptr;
    }
    
    // 创建targets列表
    PyObject* targets = PyList_New(1);
    if (!targets) {
        Py_DECREF(targetName);
        Py_DECREF(assignClass);
        handlePythonError();
        return nullptr;
    }
    
    PyList_SET_ITEM(targets, 0, targetName);  // 这里不需要DECREF targetName
    
    // 创建赋值节点的参数
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "targets", targets);
    PyDict_SetItemString(kwargs, "value", value);
    PyDict_SetItemString(kwargs, "lineno", PyLong_FromLong(1));
    PyDict_SetItemString(kwargs, "col_offset", PyLong_FromLong(0));
    PyDict_SetItemString(kwargs, "end_lineno", PyLong_FromLong(1));
    PyDict_SetItemString(kwargs, "end_col_offset", PyLong_FromLong(0));
    
    PyObject* result = PyObject_Call(assignClass, PyTuple_New(0), kwargs);
    
    Py_DECREF(assignClass);
    Py_DECREF(kwargs);
    Py_DECREF(targets);
    
    if (!result) {
        handlePythonError();
        return nullptr;
    }
    
    return result;
}

} // namespace matrix 