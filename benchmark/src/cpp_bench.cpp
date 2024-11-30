#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

void matrix_multiply(const std::vector<double>& A, 
                    const std::vector<double>& B, 
                    std::vector<double>& C, 
                    int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main() {
    std::vector<int> sizes = {64, 128, 256, 512, 1024};
    
    for (int size : sizes) {
        // 创建随机矩阵
        std::vector<double> A(size * size);
        std::vector<double> B(size * size);
        std::vector<double> C(size * size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        
        for (int i = 0; i < size * size; i++) {
            A[i] = dis(gen);
            B[i] = dis(gen);
        }
        
        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();
        
        // 执行矩阵乘法
        matrix_multiply(A, B, C, size);
        
        // 记录结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // 输出结果
        std::cout << "matrix_multiplication,C++," << size << "," 
                  << duration.count() << std::endl;
    }
    
    return 0;
}
