#include "frontend/PythonASTParser.h"
#include <iostream>

namespace boas {

PythonASTParser::PythonASTParser() 
    : initialized(false)
    , astModule(nullptr)
    , builtinsModule(nullptr)
    , builder(std::make_unique<python::PythonASTBuilder>())
    , converter(std::make_unique<python::BoasASTConverter>()) {
}

PythonASTParser::~PythonASTParser() {
    cleanup();
}

void PythonASTParser::cleanup() {
    if (astModule) {
        Py_DECREF(astModule);
        astModule = nullptr;
    }
    if (builtinsModule) {
        Py_DECREF(builtinsModule);
        builtinsModule = nullptr;
    }
    initialized = false;
}

bool PythonASTParser::initialize() {
    if (initialized) {
        return true;
    }

    // 初始化Python解释器
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    // 导入ast模块
    astModule = PyImport_ImportModule("ast");
    if (!astModule) {
        std::cerr << "Failed to import ast module" << std::endl;
        return false;
    }

    // 导入builtins模块
    builtinsModule = PyImport_ImportModule("builtins");
    if (!builtinsModule) {
        std::cerr << "Failed to import builtins module" << std::endl;
        Py_DECREF(astModule);
        astModule = nullptr;
        return false;
    }

    initialized = true;
    return true;
}

void PythonASTParser::finalize() {
    cleanup();
}

std::unique_ptr<python::PythonASTNode> PythonASTParser::parseSource(const std::string& source) {
    if (!initialized && !initialize()) {
        return nullptr;
    }

    // 将Python源代码直接传给builder
    return builder->buildFromSource(source);
}

std::unique_ptr<matrix::ExprAST> PythonASTParser::parse(const std::string& source) {
    auto pythonAst = parseSource(source);
    if (!pythonAst) {
        return nullptr;
    }
    
    auto boasAst = pythonAst->accept(converter.get());
    if (auto* moduleAst = dynamic_cast<matrix::ModuleAST*>(boasAst.get())) {
        std::vector<std::unique_ptr<matrix::ExprAST>> body;
        auto& srcBody = moduleAst->getBody();
        body.reserve(srcBody.size());
        for (auto& stmt : srcBody) {
            body.push_back(std::unique_ptr<matrix::ExprAST>(stmt->clone()));
        }
        return std::make_unique<matrix::ModuleASTImpl>(std::move(body));
    }
    
    // 如果不是 ModuleAST，将单个语句包装在 ModuleAST 中
    std::vector<std::unique_ptr<matrix::ExprAST>> stmts;
    stmts.push_back(std::move(boasAst));
    return std::make_unique<matrix::ModuleASTImpl>(std::move(stmts));
}

} // namespace boas
