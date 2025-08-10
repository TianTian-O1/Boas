#include "../include/benchmark.h"
#include <vector>
#include <random>
#include <memory>
#include <Accelerate/Accelerate.h>  // Mac 特有的加速框架

// 使用连续内存布局的矩阵类
class Matrix {
private:
    std::vector<float> data;
    size_t rows_, cols_;

public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data.resize(rows * cols);
    }

    float& operator()(size_t i, size_t j) {
        return data[i * cols_ + j];
    }

    const float& operator()(size_t i, size_t j) const {
        return data[i * cols_ + j];
    }

    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
};

// 使用 Accelerate Framework 的矩阵乘法实现
void matrix_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 调用 Accelerate Framework 的 SGEMM 函数
    cblas_sgemm(CblasRowMajor,    // 行主序
                CblasNoTrans,      // A 不转置
                CblasNoTrans,      // B 不转置
                A.rows(),          // M
                B.cols(),          // N
                A.cols(),          // K
                alpha,             // alpha
                A.ptr(),          // A
                A.cols(),         // lda
                B.ptr(),          // B
                B.cols(),         // ldb
                beta,             // beta
                C.ptr(),          // C
                C.cols());        // ldc
}

int main() {
    std::vector<int> sizes = {64, 128, 256, 512, 1024};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int size : sizes) {
        Benchmark::run("matrix_multiplication", "C++", size, 
            [&]() {
                // 创建矩阵
                Matrix A(size, size);
                Matrix B(size, size);
                Matrix C(size, size);

                // 初始化矩阵
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        A(i, j) = dis(gen);
                        B(i, j) = dis(gen);
                    }
                }

                // 执行矩阵乘法
                matrix_multiply(A, B, C);
            });
    }

    return 0;
}
