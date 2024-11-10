#include "boas/ast/list_ast.h"
#include "boas/ast/number_ast.h"
#include <stdexcept>

namespace boas {

ListAST::ListAST(std::vector<std::unique_ptr<AST>> elements)
    : elements_(std::move(elements)) {}

ListAST::ListAST(const std::vector<int64_t>& elements) {
    for (const auto& element : elements) {
        elements_.push_back(std::make_unique<NumberAST>(element));
    }
}

size_t ListAST::size() const {
    return elements_.size();
}

bool ListAST::empty() const {
    return elements_.empty();
}

void ListAST::append(std::unique_ptr<AST> element) {
    elements_.push_back(std::move(element));
}

void ListAST::extend(const ListAST& other) {
    for (const auto& element : other.elements_) {
        elements_.push_back(element->clone());
    }
}

std::optional<std::unique_ptr<AST>> ListAST::pop(int index) {
    if (empty()) {
        return std::nullopt;
    }
    
    if (index < 0) {
        index = elements_.size() + index;
    }
    
    if (index < 0 || static_cast<size_t>(index) >= elements_.size()) {
        return std::nullopt;
    }
    
    auto it = elements_.begin() + index;
    auto element = std::move(*it);
    elements_.erase(it);
    return element;
}

void ListAST::insert(size_t index, std::unique_ptr<AST> element) {
    if (index > elements_.size()) {
        throw std::out_of_range("Index out of range");
    }
    elements_.insert(elements_.begin() + index, std::move(element));
}

bool ListAST::remove(const AST& value) {
    for (auto it = elements_.begin(); it != elements_.end(); ++it) {
        if (*it->get() == value) {
            elements_.erase(it);
            return true;
        }
    }
    return false;
}

void ListAST::clear() {
    elements_.clear();
}

const AST* ListAST::at(size_t index) const {
    if (index >= elements_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return elements_[index].get();
}

AST* ListAST::at(size_t index) {
    if (index >= elements_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return elements_[index].get();
}

const std::vector<std::unique_ptr<AST>>& ListAST::getElements() const {
    return elements_;
}

std::unique_ptr<AST> ListAST::clone() const {
    std::vector<std::unique_ptr<AST>> clonedElements;
    for (const auto& element : elements_) {
        clonedElements.push_back(element->clone());
    }
    return std::make_unique<ListAST>(std::move(clonedElements));
}

bool ListAST::operator==(const AST& other) const {
    if (const ListAST* otherList = dynamic_cast<const ListAST*>(&other)) {
        if (elements_.size() != otherList->elements_.size()) {
            return false;
        }
        for (size_t i = 0; i < elements_.size(); ++i) {
            if (!(*elements_[i] == *otherList->elements_[i])) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// 实现列表推导式
std::unique_ptr<ListAST> ListAST::comprehension(
    std::unique_ptr<AST> expression,
    const std::string& varName,
    std::unique_ptr<ListAST> iterableList,
    std::unique_ptr<AST> condition) {
    
    auto result = std::make_unique<ListAST>();
    
    for (const auto& item : iterableList->getElements()) {
        // 如果有条件判断，先评估条件
        if (condition) {
            // TODO: 评估条件
            // if (!evaluateCondition(condition.get(), varName, item.get())) {
            //     continue;
            // }
        }
        
        // TODO: 使用expression生成新元素
        // auto newElement = evaluateExpression(expression.get(), varName, item.get());
        // result->append(std::move(newElement));
    }
    
    return result;
}

// 实现嵌套列表操作
void ListAST::appendList(std::unique_ptr<ListAST> nestedList) {
    isNested_ = true;
    elements_.push_back(std::move(nestedList));
}

const ListAST* ListAST::getNestedList(size_t index) const {
    if (index >= elements_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return dynamic_cast<const ListAST*>(elements_[index].get());
}

bool ListAST::isNested() const {
    return isNested_;
}

// 实现切片操作
std::unique_ptr<ListAST> ListAST::slice(int start, int end) const {
    auto result = std::make_unique<ListAST>();
    
    // 处理负索引
    if (start < 0) start = elements_.size() + start;
    if (end < 0) end = elements_.size() + end;
    
    // 确保索引在有效范围内
    start = std::max(0, std::min(static_cast<int>(elements_.size()), start));
    end = std::max(0, std::min(static_cast<int>(elements_.size()), end));
    
    for (int i = start; i < end; i++) {
        result->append(elements_[i]->clone());
    }
    
    return result;
}

// 实现其他操作...

} // namespace boas
