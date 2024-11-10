#ifndef BOAS_LIST_AST_H
#define BOAS_LIST_AST_H

#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include "boas/ast/ast.h"

namespace boas {

class ListAST : public AST {
public:
    static bool classof(const AST* ast) {
        return ast->getKind() == ASTKind::List;
    }
    
    ASTKind getKind() const override {
        return ASTKind::List;
    }
    
    ListAST() = default;
    explicit ListAST(std::vector<std::unique_ptr<AST>> elements);
    explicit ListAST(const std::vector<int64_t>& elements);
    
    // Implement virtual methods from AST
    std::unique_ptr<AST> clone() const override;
    bool operator==(const AST& other) const override;
    
    // 基本操作
    const std::vector<std::unique_ptr<AST>>& getElements() const;
    void addElement(std::unique_ptr<AST> element);
    size_t size() const;
    bool empty() const;
    
    // Python风格的list操作
    void append(std::unique_ptr<AST> element);
    void extend(const ListAST& other);
    std::optional<std::unique_ptr<AST>> pop(int index = -1);
    void insert(size_t index, std::unique_ptr<AST> element);
    bool remove(const AST& value);
    void clear();
    
    // 访问元素
    const AST* at(size_t index) const;
    AST* at(size_t index);
    
    // 新增列表推导式相关操作
    static std::unique_ptr<ListAST> comprehension(
        std::unique_ptr<AST> expression,
        const std::string& varName,
        std::unique_ptr<ListAST> iterableList,
        std::unique_ptr<AST> condition = nullptr);
    
    // 新增嵌套列表操作
    void appendList(std::unique_ptr<ListAST> nestedList);
    const ListAST* getNestedList(size_t index) const;
    bool isNested() const;
    
    // 新增Python风格的list操作
    std::unique_ptr<ListAST> slice(int start, int end) const;
    void reverse();
    void sort(std::function<bool(const AST*, const AST*)> comp = nullptr);
    size_t count(const AST& value) const;
    std::optional<size_t> index(const AST& value, size_t start = 0) const;
    
    // Slice related methods
    bool hasSlice() const { return has_slice_; }
    int64_t getSliceStart() const { return slice_start_; }
    int64_t getSliceEnd() const { return slice_end_; }
    int64_t getSliceStep() const { return slice_step_; }
    
    void setSlice(int64_t start, int64_t end, int64_t step = 1) {
        has_slice_ = true;
        slice_start_ = start;
        slice_end_ = end;
        slice_step_ = step;
    }

    // Index related methods
    bool hasIndex() const { return has_index_; }
    int64_t getIndex() const { return index_; }

    void setIndex(int64_t idx) {
        has_index_ = true;
        index_ = idx;
    }

    // 获取要进行索引操作的源列表
    const AST* getSourceList() const { return sourceList.get(); }
    void setSourceList(std::unique_ptr<AST> list) { sourceList = std::move(list); }

private:
    std::vector<std::unique_ptr<AST>> elements_;
    bool has_slice_ = false;
    int64_t slice_start_ = 0;
    int64_t slice_end_ = 0;
    int64_t slice_step_ = 1;
    bool isNested_ = false;
    bool has_index_ = false;
    int64_t index_ = 0;
    std::unique_ptr<AST> sourceList;  // 用于索引操作的源列表
};

} // namespace boas

#endif // BOAS_LIST_AST_H
