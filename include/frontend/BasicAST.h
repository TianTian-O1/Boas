#ifndef BOAS_BASIC_AST_H
#define BOAS_BASIC_AST_H

#include <Python.h>
#include "frontend/AST.h"
#include <memory>
#include <string>
#include <vector>

namespace boas {

// 基类定义
class SubscriptExprAST : public matrix::ExprAST {
protected:
    std::unique_ptr<matrix::ExprAST> value;
    std::unique_ptr<matrix::ExprAST> slice;
public:
    SubscriptExprAST(std::unique_ptr<matrix::ExprAST> value, std::unique_ptr<matrix::ExprAST> slice)
        : value(std::move(value)), slice(std::move(slice)) {}
    
    matrix::ExprAST* getValue() const { return value.get(); }
    matrix::ExprAST* getSlice() const { return slice.get(); }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Subscript(";
        value->dump(0);
        std::cout << "[";
        slice->dump(0);
        std::cout << "])";
    }
    
    Kind getKind() const override { return Kind::Member; }
};

class AttributeExprAST : public matrix::ExprAST {
protected:
    std::unique_ptr<matrix::ExprAST> value;
    std::string attr;
public:
    AttributeExprAST(std::unique_ptr<matrix::ExprAST> value, const std::string& attr)
        : value(std::move(value)), attr(attr) {}
    
    matrix::ExprAST* getValue() const { return value.get(); }
    const std::string& getAttr() const { return attr; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Attribute(";
        value->dump(0);
        std::cout << "." << attr << ")";
    }
    
    Kind getKind() const override { return Kind::Member; }
};

class ConstantExprAST : public matrix::ExprAST {
protected:
    PyObject* value;
public:
    ConstantExprAST(PyObject* value) : value(value) {
        Py_XINCREF(value);
    }
    
    ~ConstantExprAST() {
        Py_XDECREF(value);
    }
    
    PyObject* getValue() const { return value; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Constant(";
        if (PyLong_Check(value)) {
            std::cout << PyLong_AsLong(value);
        } else if (PyFloat_Check(value)) {
            std::cout << PyFloat_AsDouble(value);
        } else if (PyUnicode_Check(value)) {
            std::cout << "\"" << PyUnicode_AsUTF8(value) << "\"";
        } else {
            std::cout << "<unknown>";
        }
        std::cout << ")";
    }
    
    Kind getKind() const override { return Kind::Number; }
};

class StringExprAST : public matrix::ExprAST {
    std::string value;
public:
    StringExprAST(const std::string& value) : value(value) {}
    
    const std::string& getValue() const { return value; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "\"" << value << "\"";
    }
    
    Kind getKind() const override { return Kind::Str; }
    
    matrix::ExprAST* clone() const override {
        return new StringExprAST(value);
    }
};

class ListExprAST : public matrix::ExprAST {
    std::vector<std::unique_ptr<matrix::ExprAST>> elements;
public:
    ListExprAST(std::vector<std::unique_ptr<matrix::ExprAST>> elements)
        : elements(std::move(elements)) {}
    
    const std::vector<std::unique_ptr<matrix::ExprAST>>& getElements() const { return elements; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "[";
        for (size_t i = 0; i < elements.size(); ++i) {
            if (i > 0) std::cout << ", ";
            elements[i]->dump(0);
        }
        std::cout << "]";
    }
    
    Kind getKind() const override { return Kind::List; }
    
    matrix::ExprAST* clone() const override {
        std::vector<std::unique_ptr<matrix::ExprAST>> newElements;
        for (const auto& elem : elements) {
            newElements.push_back(std::unique_ptr<matrix::ExprAST>(elem->clone()));
        }
        return new ListExprAST(std::move(newElements));
    }
};

class VariableExprAST : public matrix::ExprAST {
    std::string name;
public:
    VariableExprAST(const std::string& name) : name(name) {}
    
    const std::string& getName() const { return name; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << name;
    }
    
    Kind getKind() const override { return Kind::Variable; }
    
    matrix::ExprAST* clone() const override {
        return new VariableExprAST(name);
    }
};

class MatmulExprAST : public matrix::ExprAST {
    std::unique_ptr<matrix::ExprAST> lhs;
    std::unique_ptr<matrix::ExprAST> rhs;
public:
    MatmulExprAST(std::unique_ptr<matrix::ExprAST> lhs, std::unique_ptr<matrix::ExprAST> rhs)
        : lhs(std::move(lhs)), rhs(std::move(rhs)) {}
    
    const matrix::ExprAST* getLHS() const { return lhs.get(); }
    const matrix::ExprAST* getRHS() const { return rhs.get(); }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "matmul(";
        lhs->dump(0);
        std::cout << ", ";
        rhs->dump(0);
        std::cout << ")";
    }
    
    Kind getKind() const override { return Kind::Matmul; }
    
    matrix::ExprAST* clone() const override {
        return new MatmulExprAST(
            std::unique_ptr<matrix::ExprAST>(lhs->clone()),
            std::unique_ptr<matrix::ExprAST>(rhs->clone())
        );
    }
};

} // namespace boas

#endif // BOAS_BASIC_AST_H 