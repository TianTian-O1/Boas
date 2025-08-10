#include "frontend/python_parser/PythonASTBuilder.h"
#include <iostream>

namespace boas {
namespace python {

PythonASTBuilder::PythonASTBuilder() : initialized(false), astModule(nullptr) {
    // 初始化Python解释器
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    
    // 导入ast模块
    astModule = PyImport_ImportModule("ast");
    if (!astModule) {
        std::cerr << "Failed to import ast module" << std::endl;
        PyErr_Print();
        return;
    }
    
    initialized = true;
}

PythonASTBuilder::~PythonASTBuilder() {
    if (astModule) {
        Py_DECREF(astModule);
    }
}

std::unique_ptr<ModuleNode> PythonASTBuilder::buildModule(PyObject* node) {
    // 获取模块体
    PyObject* body = PyObject_GetAttrString(node, "body");
    if (!body || !PyList_Check(body)) {
        Py_XDECREF(body);
        return nullptr;
    }

    // 处理模块中的所有语句
    std::vector<std::unique_ptr<PythonASTNode>> statements;
    Py_ssize_t size = PyList_Size(body);
    
    // 调试输出
    std::cout << "Module body size: " << size << std::endl;
    
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(body, i);
        
        // 获取语句类型
        PyObject* cls = PyObject_GetAttrString(item, "__class__");
        PyObject* name = PyObject_GetAttrString(cls, "__name__");
        std::string className = PyUnicode_AsUTF8(name);
        std::cout << "Processing statement " << i << " of type: " << className << std::endl;
        Py_DECREF(cls);
        Py_DECREF(name);
        
        if (auto stmt = buildFromPyObject(item)) {
            statements.push_back(std::move(stmt));
        } else {
            std::cout << "Failed to build statement " << i << std::endl;
        }
    }

    Py_DECREF(body);

    // 创建并返回模块节点
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
    // 获取表达式的值
    PyObject* value = PyObject_GetAttrString(node, "value");
    if (!value) {
        return nullptr;
    }

    // 构建表达式值节点
    auto exprValue = buildFromPyObject(value);
    Py_DECREF(value);

    if (!exprValue) {
        return nullptr;
    }

    // 创建并返回表达式节点
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
    // 获取常量值
    PyObject* value = PyObject_GetAttrString(node, "value");
    if (!value) {
        return nullptr;
    }

    // 根据值的类型创建相应的节点
    if (PyFloat_Check(value) || PyLong_Check(value)) {
        // 数值类型
        double numValue = PyFloat_AsDouble(value);
        Py_DECREF(value);
        return std::make_unique<NumNode>(numValue);
    } else if (PyUnicode_Check(value)) {
        // 字符串类型
        std::string strValue = PyUnicode_AsUTF8(value);
        Py_DECREF(value);
        return std::make_unique<StringNode>(strValue);
    } else if (PyBool_Check(value)) {
        // 布尔类型，转换为数值
        double boolValue = (value == Py_True) ? 1.0 : 0.0;
        Py_DECREF(value);
        return std::make_unique<NumNode>(boolValue);
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
    // 获取赋值目标
    PyObject* targets = PyObject_GetAttrString(node, "targets");
    if (!targets || !PyList_Check(targets)) {
        Py_XDECREF(targets);
        return nullptr;
    }

    // 目前只处理第一个赋值目标
    if (PyList_Size(targets) == 0) {
        Py_DECREF(targets);
        return nullptr;
    }

    PyObject* target = PyList_GetItem(targets, 0);
    std::string targetName;

    // 处理简单的变量名赋值
    if (PyObject_HasAttrString(target, "id")) {
        PyObject* targetId = PyObject_GetAttrString(target, "id");
        if (targetId && PyUnicode_Check(targetId)) {
            targetName = PyUnicode_AsUTF8(targetId);
            Py_DECREF(targetId);
        }
    }

    Py_DECREF(targets);

    if (targetName.empty()) {
        return nullptr;
    }

    // 获取赋值的值
    PyObject* value = PyObject_GetAttrString(node, "value");
    if (!value) {
        return nullptr;
    }

    // 构建值节点
    auto valueNode = buildFromPyObject(value);
    Py_DECREF(value);

    if (!valueNode) {
        return nullptr;
    }

    // 创建并返回赋值节点
    return std::make_unique<AssignNode>(targetName, std::move(valueNode));
}

std::unique_ptr<CallNode> PythonASTBuilder::buildCall(PyObject* node) {
    // 检查是否是方法调用
    PyObject* func = PyObject_GetAttrString(node, "func");
    if (!func) {
        std::cerr << "Failed to get func attribute in buildCall" << std::endl;
        return nullptr;
    }

    // 如果func是Attribute节点，则应该作为方法调用处理
    PyObject* funcType = PyObject_GetAttrString(func, "__class__");
    PyObject* funcTypeName = PyObject_GetAttrString(funcType, "__name__");
    std::string className = PyUnicode_AsUTF8(funcTypeName);
    Py_DECREF(funcType);
    Py_DECREF(funcTypeName);

    if (className == "Attribute") {
        Py_DECREF(func);
        if (auto methodCall = buildMethodCall(node)) {
            // 将MethodCallNode转换为CallNode
            std::vector<std::unique_ptr<PythonASTNode>> args;
            args.push_back(std::move(methodCall));
            return std::make_unique<CallNode>("to", std::move(args));
        }
        return nullptr;
    }

    // 处理普通函数调用
    std::string funcName;
    if (PyObject_HasAttrString(func, "id")) {
        PyObject* id = PyObject_GetAttrString(func, "id");
        if (id && PyUnicode_Check(id)) {
            funcName = PyUnicode_AsUTF8(id);
            Py_DECREF(id);
        }
    }

    Py_DECREF(func);

    if (funcName.empty()) {
        std::cerr << "Failed to get function name" << std::endl;
        return nullptr;
    }

    // 获取参数列表
    PyObject* args = PyObject_GetAttrString(node, "args");
    if (!args || !PyList_Check(args)) {
        std::cerr << "Failed to get args in buildCall" << std::endl;
        Py_XDECREF(args);
        return nullptr;
    }

    // 处理参数
    std::vector<std::unique_ptr<PythonASTNode>> argNodes;
    Py_ssize_t size = PyList_Size(args);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* arg = PyList_GetItem(args, i);
        if (auto argNode = buildFromPyObject(arg)) {
            argNodes.push_back(std::move(argNode));
        } else {
            std::cerr << "Failed to build argument " << i << " in buildCall" << std::endl;
        }
    }

    Py_DECREF(args);

    return std::make_unique<CallNode>(funcName, std::move(argNodes));
}

std::unique_ptr<PythonASTNode> PythonASTBuilder::buildFromPyObject(PyObject* node) {
    if (!node || !PyObject_HasAttrString(node, "__class__")) {
        return nullptr;
    }

    PyObject* cls = PyObject_GetAttrString(node, "__class__");
    PyObject* name = PyObject_GetAttrString(cls, "__name__");
    if (!name || !PyUnicode_Check(name)) {
        Py_XDECREF(cls);
        Py_XDECREF(name);
        return nullptr;
    }

    std::string className = PyUnicode_AsUTF8(name);
    Py_DECREF(name);
    Py_DECREF(cls);

    if (className == "Module") {
        return buildModule(node);
    } else if (className == "FunctionDef") {
        return buildFunctionDef(node);
    } else if (className == "Expr") {
        return buildExpr(node);
    } else if (className == "BinOp") {
        return buildBinOp(node);
    } else if (className == "Constant") {
        return buildConstant(node);
    } else if (className == "Num") {
        return buildNum(node);
    } else if (className == "Name") {
        return buildName(node);
    } else if (className == "Assign") {
        return buildAssign(node);
    } else if (className == "Call") {
        // 检查是否是方法调用
        if (PyObject_HasAttrString(node, "func") && 
            PyObject_HasAttrString(PyObject_GetAttrString(node, "func"), "value")) {
            return buildMethodCall(node);
        }
        return buildCall(node);
    } else if (className == "List") {
        return buildList(node);
    } else if (className == "For") {
        return buildFor(node);
    } else if (className == "Return") {
        return buildReturn(node);
    } else if (className == "Attribute") {
        return buildAttribute(node);
    } else if (className == "Import") {
        return buildImport(node);
    }

    return nullptr;
}

std::unique_ptr<PythonASTNode> PythonASTBuilder::buildFromSource(const std::string& source) {
    if (!initialized) {
        std::cerr << "PythonASTBuilder not initialized" << std::endl;
        return nullptr;
    }

    // 获取ast.parse函数
    PyObject* parseFunc = PyObject_GetAttrString(astModule, "parse");
    if (!parseFunc) {
        std::cerr << "Failed to get ast.parse function" << std::endl;
        PyErr_Print();
        return nullptr;
    }

    // 将源代码转换为Python字符串
    PyObject* sourceStr = PyUnicode_FromString(source.c_str());
    if (!sourceStr) {
        std::cerr << "Failed to convert source to Python string" << std::endl;
        Py_DECREF(parseFunc);
        PyErr_Print();
        return nullptr;
    }

    // 调用ast.parse函数
    PyObject* args = PyTuple_Pack(1, sourceStr);
    PyObject* ast = PyObject_CallObject(parseFunc, args);
    
    Py_DECREF(sourceStr);
    Py_DECREF(args);
    Py_DECREF(parseFunc);

    if (!ast) {
        std::cerr << "Failed to parse source code" << std::endl;
        PyErr_Print();
        return nullptr;
    }

    // 打印AST的类型和属性
    PyObject* astType = PyObject_GetAttrString(ast, "__class__");
    PyObject* typeName = PyObject_GetAttrString(astType, "__name__");
    std::cout << "AST type: " << PyUnicode_AsUTF8(typeName) << std::endl;
    
    // 打印AST的body属性
    PyObject* body = PyObject_GetAttrString(ast, "body");
    if (PyList_Check(body)) {
        Py_ssize_t size = PyList_Size(body);
        std::cout << "AST body size: " << size << std::endl;
        
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* item = PyList_GetItem(body, i);
            PyObject* itemType = PyObject_GetAttrString(item, "__class__");
            PyObject* itemTypeName = PyObject_GetAttrString(itemType, "__name__");
            std::cout << "Body item " << i << " type: " << PyUnicode_AsUTF8(itemTypeName) << std::endl;
            Py_DECREF(itemType);
            Py_DECREF(itemTypeName);
        }
    }
    
    Py_DECREF(astType);
    Py_DECREF(typeName);
    Py_DECREF(body);

    // 构建AST节点
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

// 新增：构建属性访问节点
std::unique_ptr<AttributeNode> PythonASTBuilder::buildAttribute(PyObject* node) {
    PyObject* value = PyObject_GetAttrString(node, "value");
    if (!value) {
        return nullptr;
    }

    PyObject* attr = PyObject_GetAttrString(node, "attr");
    if (!attr || !PyUnicode_Check(attr)) {
        Py_XDECREF(value);
        Py_XDECREF(attr);
        return nullptr;
    }

    auto valueNode = buildFromPyObject(value);
    std::string attrName = PyUnicode_AsUTF8(attr);

    Py_DECREF(value);
    Py_DECREF(attr);

    if (!valueNode) {
        return nullptr;
    }

    return std::make_unique<AttributeNode>(std::move(valueNode), attrName);
}

// 新增：构建方法调用节点
std::unique_ptr<MethodCallNode> PythonASTBuilder::buildMethodCall(PyObject* node) {
    // 获取函数对象
    PyObject* func = PyObject_GetAttrString(node, "func");
    if (!func) {
        std::cerr << "Failed to get func attribute" << std::endl;
        return nullptr;
    }

    // 检查是否是属性访问
    if (!PyObject_HasAttrString(func, "value") || !PyObject_HasAttrString(func, "attr")) {
        std::cerr << "Not an attribute access" << std::endl;
        Py_DECREF(func);
        return nullptr;
    }

    // 获取对象和方法名
    PyObject* value = PyObject_GetAttrString(func, "value");
    PyObject* attr = PyObject_GetAttrString(func, "attr");
    
    if (!value || !attr || !PyUnicode_Check(attr)) {
        std::cerr << "Failed to get value or attr" << std::endl;
        Py_XDECREF(func);
        Py_XDECREF(value);
        Py_XDECREF(attr);
        return nullptr;
    }

    // 构建参数列表
    PyObject* args = PyObject_GetAttrString(node, "args");
    if (!args || !PyList_Check(args)) {
        std::cerr << "Failed to get args" << std::endl;
        Py_XDECREF(func);
        Py_XDECREF(value);
        Py_XDECREF(attr);
        Py_XDECREF(args);
        return nullptr;
    }

    // 递归构建对象节点（可能是另一个方法调用）
    auto valueNode = buildFromPyObject(value);
    std::string methodName = PyUnicode_AsUTF8(attr);

    std::cout << "Building method call: " << methodName << std::endl;

    // 构建参数节点列表
    std::vector<std::unique_ptr<PythonASTNode>> argNodes;
    Py_ssize_t size = PyList_Size(args);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* arg = PyList_GetItem(args, i);
        if (auto argNode = buildFromPyObject(arg)) {
            argNodes.push_back(std::move(argNode));
        } else {
            std::cerr << "Failed to build argument " << i << std::endl;
        }
    }

    Py_DECREF(func);
    Py_DECREF(value);
    Py_DECREF(attr);
    Py_DECREF(args);

    if (!valueNode) {
        std::cerr << "Failed to build value node" << std::endl;
        return nullptr;
    }

    return std::make_unique<MethodCallNode>(std::move(valueNode), methodName, std::move(argNodes));
}

std::unique_ptr<PythonASTNode> PythonASTBuilder::buildImport(PyObject* node) {
    PyObject* names = PyObject_GetAttrString(node, "names");
    if (!names || !PyList_Check(names)) {
        Py_XDECREF(names);
        return nullptr;
    }

    // 目前只处理第一个导入名称
    if (PyList_Size(names) > 0) {
        PyObject* alias = PyList_GetItem(names, 0);
        PyObject* name = PyObject_GetAttrString(alias, "name");
        if (name && PyUnicode_Check(name)) {
            std::string importName = PyUnicode_AsUTF8(name);
            Py_DECREF(name);
            Py_DECREF(names);
            return std::make_unique<ImportNode>(importName);
        }
        Py_XDECREF(name);
    }

    Py_DECREF(names);
    return nullptr;
}

} // namespace python
} // namespace boas 