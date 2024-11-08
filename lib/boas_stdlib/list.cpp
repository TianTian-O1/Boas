#include <vector>
#include <cstdint>
#include <stdexcept>

// 列表结构体
struct List {
    std::vector<double> elements;
};

extern "C" {

// 创建列表
void* createList(int64_t size) {
    auto list = new List();
    list->elements.reserve(size);
    return list;
}

// 获取列表元素
double listGetItem(void* list, int64_t index) {
    auto l = static_cast<List*>(list);
    if (index < 0) {
        index = static_cast<int64_t>(l->elements.size()) + index;
    }
    if (index >= 0 && static_cast<size_t>(index) < l->elements.size()) {
        return l->elements[index];
    }
    throw std::out_of_range("Index out of range");
}

// 添加元素
void listAppend(void* list, double value) {
    auto l = static_cast<List*>(list);
    l->elements.push_back(value);
}

// 弹出最后一个元素
double listPop(void* list) {
    auto l = static_cast<List*>(list);
    if (l->elements.empty()) {
        throw std::out_of_range("Cannot pop from empty list");
    }
    double value = l->elements.back();
    l->elements.pop_back();
    return value;
}

// 获取列表大小
int64_t listSize(void* list) {
    auto l = static_cast<List*>(list);
    return static_cast<int64_t>(l->elements.size());
}

// 列表切片
void* listSlice(void* list, int64_t start, int64_t end) {
    auto l = static_cast<List*>(list);
    auto newList = new List();
    
    if (start < 0) start = static_cast<int64_t>(l->elements.size()) + start;
    if (end < 0) end = static_cast<int64_t>(l->elements.size()) + end;
    
    start = std::max(int64_t(0), std::min(start, static_cast<int64_t>(l->elements.size())));
    end = std::max(int64_t(0), std::min(end, static_cast<int64_t>(l->elements.size())));
    
    for (int64_t i = start; i < end; ++i) {
        newList->elements.push_back(l->elements[i]);
    }
    return newList;
}

// 删除列表
void deleteList(void* list) {
    delete static_cast<List*>(list);
}

} 