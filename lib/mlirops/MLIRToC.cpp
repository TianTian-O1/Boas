// lib/mlirops/MLIRToC.cpp
#include "mlirops/MLIRToC.h"
#include <sstream>
#include <regex>
#include <iostream>
#include <map>
#include <vector>
#include <string>

namespace matrix {

std::vector<MatrixInfo> MLIRToC::parseMatrices(const std::string& mlirInput) {
    std::vector<MatrixInfo> matrices;
    
    // 首先找到所有矩阵分配
    std::regex alloc_pattern(R"(%alloc(_\d*)? = memref.alloc\(\) : memref<(\d+)x(\d+)xf64>)");
    std::string::const_iterator searchStart(mlirInput.cbegin());
    std::smatch matches;
    
    while (std::regex_search(searchStart, mlirInput.cend(), matches, alloc_pattern)) {
        MatrixInfo matrix;
        matrix.name = matches[1].str().empty() ? "alloc" : "alloc" + matches[1].str();
        matrix.rows = std::stoi(matches[2]);
        matrix.cols = std::stoi(matches[3]);
        matrix.values.resize(matrix.rows * matrix.cols, 0.0);
        matrices.push_back(matrix);
        searchStart = matches.suffix().first;
    }

    // 解析存储操作
    std::regex store_pattern(R"(memref\.store %cst(?:_\d*)? = arith\.constant\s+([\d.]+).*?%alloc(_\d*)?\[%c(\d+)(?:_\d+)?, %c(\d+)(?:_\d+)?\])");
    searchStart = mlirInput.cbegin();
    
    while (std::regex_search(searchStart, mlirInput.cend(), matches, store_pattern)) {
        double value = std::stod(matches[1]);
        std::string allocName = matches[2].str().empty() ? "alloc" : "alloc" + matches[2].str();
        int row = std::stoi(matches[3]);
        int col = std::stoi(matches[4]);
        
        // 找到对应的矩阵并设置值
        for (auto& matrix : matrices) {
            if (matrix.name == allocName) {
                size_t index = row * matrix.cols + col;
                if (index < matrix.values.size()) {
                    matrix.values[index] = value;
                }
                break;
            }
        }
        
        searchStart = matches.suffix().first;
    }
    
    // 调试输出
    std::cerr << "\nParsed matrices:\n";
    for (const auto& matrix : matrices) {
        std::cerr << "Matrix " << matrix.name << " (" << matrix.rows << "x" << matrix.cols << "):\n";
        for (int i = 0; i < matrix.rows; i++) {
            std::cerr << "[";
            for (int j = 0; j < matrix.cols; j++) {
                std::cerr << " " << matrix.values[i * matrix.cols + j];
            }
            std::cerr << " ]\n";
        }
    }

    return matrices;
}

std::string MLIRToC::convertToC(const std::string& mlirInput) {
    auto matrices = parseMatrices(mlirInput);
    std::stringstream ss;
    
    // 基本的类型和函数定义
    ss << "#include <stdio.h>\n"
       << "#include <stdlib.h>\n\n"
       << "typedef struct { int rows, cols; double* data; } Matrix;\n\n"
       << "Matrix* create_matrix(int rows, int cols, const double* data) {\n"
       << "    Matrix* m = malloc(sizeof(Matrix));\n"
       << "    m->rows = rows; m->cols = cols;\n"
       << "    m->data = malloc(rows * cols * sizeof(double));\n"
       << "    if(data) for(int i = 0; i < rows * cols; i++) m->data[i] = data[i];\n"
       << "    return m;\n"
       << "}\n\n"
       << "Matrix* matmul(Matrix* A, Matrix* B) {\n"
       << "    Matrix* C = create_matrix(A->rows, B->cols, NULL);\n"
       << "    for(int i = 0; i < A->rows; i++)\n"
       << "        for(int j = 0; j < B->cols; j++) {\n"
       << "            double sum = 0;\n"
       << "            for(int k = 0; k < A->cols; k++)\n"
       << "                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];\n"
       << "            C->data[i * C->cols + j] = sum;\n"
       << "        }\n"
       << "    return C;\n"
       << "}\n\n"
       << "int main() {\n";

    // 生成矩阵定义
    for (size_t i = 0; i < matrices.size(); i++) {
        const auto& matrix = matrices[i];
        std::string name = "mat" + std::to_string(i);
        
        ss << "    double data" << i << "[] = {";
        for (size_t j = 0; j < matrix.values.size(); j++) {
            if (j > 0) ss << ",";
            ss << matrix.values[j];
        }
        ss << "};\n";
        ss << "    Matrix* " << name << " = create_matrix("
           << matrix.rows << ", " << matrix.cols << ", data" << i << ");\n";
    }

    // 生成矩阵乘法和输出
    if (matrices.size() >= 2) {
        ss << "\n    Matrix* result = matmul(mat0, mat1);\n"
           << "    for(int i = 0; i < result->rows; i++) {\n"
           << "        for(int j = 0; j < result->cols; j++)\n"
           << "            printf(\"%g \", result->data[i * result->cols + j]);\n"
           << "        printf(\"\\n\");\n"
           << "    }\n";
    }

    ss << "    return 0;\n}\n";
    
    return ss.str();
}



} // namespace matrix