#include "frontend/python_parser/PythonASTBuilder.h"
#include <iostream>

namespace boas {
namespace python {

PythonASTBuilder::PythonASTBuilder() : initialized(false), astModule(nullptr) {}

PythonASTBuilder::~PythonASTBuilder() {
    if (astModule) {
        Py_DECREF(astModule);
    }
}

std::unique_ptr<ModuleNode> PythonASTBuilder::buildModule(PyObject* node) {
    if (!PyObject_HasAttrString(node, "body")) {
        return nullptr;
    }

    PyObject* body = PyObject_GetAttrString(node, "body");
    if (!body || !PyList_Check(body)) {
        Py_XDECREF(body);
        return nullptr;
    }

    std::vector<std::unique_ptr<PythonASTNode>> statements;
    Py_ssize_t size = PyList_Size(body);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(body, i);
        if (auto stmt = buildFromPyObject(item)) {
            statements.push_back(std::move(stmt));
        }
    }

    Py_DECREF(body);
    return std::make_unique<ModuleNode>(std::move(statements));
}

std::unique_ptr<FunctionDefNode> PythonASTBuilder::buildFunctionDef(PyObject* node) {
    PyObject* nameObj = PyObject_GetAttrString(node, "name");
    if (!nameObj || !PyUnicode_Check(nameObj)) {
        Py_XDECREF(nameObj);
        return nullptr;
    }
    std::string name = PyUnicode_AsUTF8(nameObj);
    Py_DECREF(nameObj);

    PyObject* argsObj = PyObject_GetAttrString(node, "args");
    if (!argsObj) {
        return nullptr;
    }

    PyObject* argsList = PyObject_GetAttrString(argsObj, "args");
    if (!argsList || !PyList_Check(argsList)) {
        Py_XDECREF(argsObj);
        Py_XDECREF(argsList);
        return nullptr;
    }

    std::vector<std::string> args;
    Py_ssize_t argsSize = PyList_Size(argsList);
    for (Py_ssize_t i = 0; i < argsSize; i++) {
        PyObject* arg = PyList_GetItem(argsList, i);
        PyObject* argName = PyObject_GetAttrString(arg, "arg");
        if (argName && PyUnicode_Check(argName)) {
            args.push_back(PyUnicode_AsUTF8(argName));
        }
        Py_XDECREF(argName);
    }

    Py_DECREF(argsList);
    Py_DECREF(argsObj);

    PyObject* body = PyObject_GetAttrString(node, "body");
    if (!body || !PyList_Check(body)) {
        Py_XDECREF(body);
        return nullptr;
    }

    std::vector<std::unique_ptr<PythonASTNode>> statements;
    Py_ssize_t size = PyList_Size(body);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(body, i);
        if (auto stmt = buildFromPyObject(item)) {
            statements.push_back(std::move(stmt));
        }
    }

    Py_DECREF(body);
    return std::make_unique<FunctionDefNode>(name, std::move(args), std::move(statements));
}

std::unique_ptr<ExprNode> PythonASTBuilder::buildExpr(PyObject* node) {
    PyObject* value = PyObject_GetAttrString(node, "value");
    if (!value) {
        return nullptr;
    }

    auto exprValue = buildFromPyObject(value);
    Py_DECREF(value);

    if (!exprValue) {
        return nullptr;
    }

    return std::make_unique<ExprNode>(std::move(exprValue));
}

std::unique_ptr<BinOpNode> PythonASTBuilder::buildBinOp(PyObject* node) {
    PyObject* left = PyObject_GetAttrString(node, "left");
    PyObject* right = PyObject_GetAttrString(node, "right");
    PyObject* op = PyObject_GetAttrString(node, "op");

    if (!left || !right || !op) {
        Py_XDECREF(left);
        Py_XDECREF(right);
        Py_XDECREF(op);
        return nullptr;
    }

    auto leftNode = buildFromPyObject(left);
    auto rightNode = buildFromPyObject(right);

    std::string opStr;
    if (PyObject_HasAttrString(op, "__class__")) {
        PyObject* opClass = PyObject_GetAttrString(op, "__class__");
        PyObject* opName = PyObject_GetAttrString(opClass, "__name__");
        if (opName && PyUnicode_Check(opName)) {
            opStr = PyUnicode_AsUTF8(opName);
        }
        Py_XDECREF(opClass);
        Py_XDECREF(opName);
    }

    Py_DECREF(left);
    Py_DECREF(right);
    Py_DECREF(op);

    if (!leftNode || !rightNode) {
        return nullptr;
    }

    std::string opSymbol;
    if (opStr == "Add") opSymbol = "+";
    else if (opStr == "Sub") opSymbol = "-";
    else if (opStr == "Mult") opSymbol = "*";
    else if (opStr == "Div") opSymbol = "/";
    else if (opStr == "MatMult") opSymbol = "@";
    else opSymbol = "?";

    return std::make_unique<BinOpNode>(
        std::move(leftNode),
        std::move(rightNode),
        opSymbol
    );
}

std::unique_ptr<PythonASTNode> PythonASTBuilder::buildConstant(PyObject* node) {
    if (!PyObject_HasAttrString(node, "value")) {
        return nullptr;
    }

    PyObject* value = PyObject_GetAttrString(node, "value");
    if (!value) {
        return nullptr;
    }

    if (PyFloat_Check(value) || PyLong_Check(value)) {
        double numValue = PyFloat_AsDouble(value);
        Py_DECREF(value);
        return std::make_unique<NumNode>(numValue);
    } else if (PyUnicode_Check(value)) {
        std::string strValue = PyUnicode_AsUTF8(value);
        Py_DECREF(value);
        return std::make_unique<StringNode>(strValue);
    }
    
    Py_DECREF(value);
    return nullptr;
}

std::unique_ptr<NumNode> PythonASTBuilder::buildNum(PyObject* node) {
    if (!PyObject_HasAttrString(node, "value")) {
        return nullptr;
    }

    PyObject* value = PyObject_GetAttrString(node, "value");
    if (!value) {
        return nullptr;
    }

    if (PyFloat_Check(value) || PyLong_Check(value)) {
        double numValue = PyFloat_AsDouble(value);
        Py_DECREF(value);
        return std::make_unique<NumNode>(numValue);
    }
    
    Py_DECREF(value);
    return nullptr;
}

std::unique_ptr<NameNode> PythonASTBuilder::buildName(PyObject* node) {
    PyObject* id = PyObject_GetAttrString(node, "id");
    if (!id || !PyUnicode_Check(id)) {
        Py_XDECREF(id);
        return nullptr;
    }

    std::string name = PyUnicode_AsUTF8(id);
    Py_DECREF(id);

    return std::make_unique<NameNode>(name);
}

std::unique_ptr<AssignNode> PythonASTBuilder::buildAssign(PyObject* node) {
    PyObject* targets = PyObject_GetAttrString(node, "targets");
    PyObject* value = PyObject_GetAttrString(node, "value");

    if (!targets || !PyList_Check(targets) || !value) {
        Py_XDECREF(targets);
        Py_XDECREF(value);
        return nullptr;
    }

    if (PyList_Size(targets) == 0) {
        Py_DECREF(targets);
        Py_DECREF(value);
        return nullptr;
    }

    PyObject* target = PyList_GetItem(targets, 0);
    PyObject* targetId = PyObject_GetAttrString(target, "id");
    if (!targetId || !PyUnicode_Check(targetId)) {
        Py_DECREF(targets);
        Py_DECREF(value);
        Py_XDECREF(targetId);
        return nullptr;
    }

    std::string targetName = PyUnicode_AsUTF8(targetId);
    auto valueNode = buildFromPyObject(value);

    Py_DECREF(targets);
    Py_DECREF(value);
    Py_DECREF(targetId);

    if (!valueNode) {
        return nullptr;
    }

    return std::make_unique<AssignNode>(targetName, std::move(valueNode));
}

std::unique_ptr<CallNode> PythonASTBuilder::buildCall(PyObject* node) {
    PyObject* funcObj = PyObject_GetAttrString(node, "func");
    if (!funcObj) {
        return nullptr;
    }

    std::string funcName;
    PyObject* attrClass = PyObject_GetAttrString(astModule, "Attribute");
    if (PyObject_IsInstance(funcObj, attrClass)) {
        // 属性访问，比如 tensor.matmul
        PyObject* valueObj = PyObject_GetAttrString(funcObj, "value");
        PyObject* attrObj = PyObject_GetAttrString(funcObj, "attr");
        
        if (valueObj && attrObj && PyUnicode_Check(attrObj)) {
            PyObject* moduleNameObj = PyObject_GetAttrString(valueObj, "id");
            if (moduleNameObj && PyUnicode_Check(moduleNameObj)) {
                funcName = std::string(PyUnicode_AsUTF8(moduleNameObj)) + "." + 
                          std::string(PyUnicode_AsUTF8(attrObj));
            }
            Py_XDECREF(moduleNameObj);
        }
        
        Py_XDECREF(valueObj);
        Py_XDECREF(attrObj);
    } else if (PyObject_HasAttrString(funcObj, "id")) {
        // 普通函数调用
        PyObject* nameObj = PyObject_GetAttrString(funcObj, "id");
        if (nameObj && PyUnicode_Check(nameObj)) {
            funcName = PyUnicode_AsUTF8(nameObj);
        }
        Py_XDECREF(nameObj);
    }
    
    Py_XDECREF(attrClass);
    Py_DECREF(funcObj);

    if (funcName.empty()) {
        return nullptr;
    }

    PyObject* argsObj = PyObject_GetAttrString(node, "args");
    if (!argsObj || !PyList_Check(argsObj)) {
        Py_XDECREF(argsObj);
        return nullptr;
    }

    std::vector<std::unique_ptr<PythonASTNode>> args;
    Py_ssize_t size = PyList_Size(argsObj);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* arg = PyList_GetItem(argsObj, i);
        if (auto argNode = buildFromPyObject(arg)) {
            args.push_back(std::move(argNode));
        }
    }

    Py_DECREF(argsObj);
    return std::make_unique<CallNode>(funcName, std::move(args));
}

std::unique_ptr<PythonASTNode> PythonASTBuilder::buildFromPyObject(PyObject* node) {
    if (!node) return nullptr;

    PyObject* typeObj = PyObject_GetAttrString(node, "__class__");
    if (!typeObj) return nullptr;

    PyObject* typeName = PyObject_GetAttrString(typeObj, "__name__");
    Py_DECREF(typeObj);

    if (!typeName || !PyUnicode_Check(typeName)) {
        Py_XDECREF(typeName);
        return nullptr;
    }

    std::string type = PyUnicode_AsUTF8(typeName);
    Py_DECREF(typeName);

    if (type == "Module") {
        return buildModule(node);
    } else if (type == "FunctionDef") {
        return buildFunctionDef(node);
    } else if (type == "Expr") {
        return buildExpr(node);
    } else if (type == "BinOp") {
        return buildBinOp(node);
    } else if (type == "Num" || type == "Constant") {
        return buildConstant(node);
    } else if (type == "Name") {
        return buildName(node);
    } else if (type == "Assign") {
        return buildAssign(node);
    } else if (type == "Call") {
        return buildCall(node);
    } else if (type == "List") {
        return buildList(node);
    } else if (type == "For") {
        return buildFor(node);
    } else if (type == "Return") {
        return buildReturn(node);
    }

    return nullptr;
}

std::unique_ptr<PythonASTNode> PythonASTBuilder::buildFromSource(const std::string& source) {
    if (!initialized) {
        astModule = PyImport_ImportModule("ast");
        if (!astModule) {
            return nullptr;
        }
        initialized = true;
    }

    PyObject* parseFunc = PyObject_GetAttrString(astModule, "parse");
    if (!parseFunc) {
        return nullptr;
    }

    PyObject* sourceStr = PyUnicode_FromString(source.c_str());
    if (!sourceStr) {
        Py_DECREF(parseFunc);
        return nullptr;
    }

    PyObject* ast = PyObject_CallFunctionObjArgs(parseFunc, sourceStr, NULL);
    Py_DECREF(parseFunc);
    Py_DECREF(sourceStr);

    if (!ast) {
        return nullptr;
    }

    auto result = buildFromPyObject(ast);
    Py_DECREF(ast);
    return result;
}

std::unique_ptr<ListNode> PythonASTBuilder::buildList(PyObject* node) {
    PyObject* elts = PyObject_GetAttrString(node, "elts");
    if (!elts || !PyList_Check(elts)) {
        Py_XDECREF(elts);
        return nullptr;
    }

    std::vector<std::unique_ptr<PythonASTNode>> elements;
    Py_ssize_t size = PyList_Size(elts);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(elts, i);
        if (auto element = buildFromPyObject(item)) {
            elements.push_back(std::move(element));
        }
    }

    Py_DECREF(elts);
    return std::make_unique<ListNode>(std::move(elements));
}

std::unique_ptr<ForNode> PythonASTBuilder::buildFor(PyObject* node) {
    PyObject* targetObj = PyObject_GetAttrString(node, "target");
    if (!targetObj) return nullptr;

    PyObject* targetId = PyObject_GetAttrString(targetObj, "id");
    if (!targetId || !PyUnicode_Check(targetId)) {
        Py_XDECREF(targetObj);
        Py_XDECREF(targetId);
        return nullptr;
    }
    std::string target = PyUnicode_AsUTF8(targetId);
    Py_DECREF(targetId);
    Py_DECREF(targetObj);

    PyObject* iterObj = PyObject_GetAttrString(node, "iter");
    if (!iterObj) return nullptr;
    auto iter = buildFromPyObject(iterObj);
    Py_DECREF(iterObj);
    if (!iter) return nullptr;

    PyObject* body = PyObject_GetAttrString(node, "body");
    if (!body || !PyList_Check(body)) {
        Py_XDECREF(body);
        return nullptr;
    }

    std::vector<std::unique_ptr<PythonASTNode>> statements;
    Py_ssize_t size = PyList_Size(body);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(body, i);
        if (auto stmt = buildFromPyObject(item)) {
            statements.push_back(std::move(stmt));
        }
    }
    Py_DECREF(body);

    return std::make_unique<ForNode>(target, std::move(iter), std::move(statements));
}

std::unique_ptr<ReturnNode> PythonASTBuilder::buildReturn(PyObject* node) {
    PyObject* value = PyObject_GetAttrString(node, "value");
    if (!value) return nullptr;

    auto returnValue = buildFromPyObject(value);
    Py_DECREF(value);

    if (!returnValue) return nullptr;

    return std::make_unique<ReturnNode>(std::move(returnValue));
}

} // namespace python
} // namespace boas 