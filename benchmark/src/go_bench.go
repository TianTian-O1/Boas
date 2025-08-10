package main

import (
    "fmt"
    "math/rand"
    "os"
    "time"
    "encoding/csv"
    "strconv"
    "runtime"
    "sync"
)

// 使用分块并行算法的矩阵乘法
func matrixMultiply(A, B [][]float64) [][]float64 {
    n := len(A)
    C := make([][]float64, n)
    for i := range C {
        C[i] = make([]float64, n)
    }

    // 设置最佳块大小和并行度
    blockSize := 32
    numWorkers := runtime.NumCPU()
    var wg sync.WaitGroup

    // 创建工作通道
    jobs := make(chan int, n)
    
    // 启动工作协程
    wg.Add(numWorkers)
    for w := 0; w < numWorkers; w++ {
        go func() {
            defer wg.Done()
            for i := range jobs {
                for j0 := 0; j0 < n; j0 += blockSize {
                    jEnd := min(j0+blockSize, n)
                    for k0 := 0; k0 < n; k0 += blockSize {
                        kEnd := min(k0+blockSize, n)
                        
                        // 计算当前块
                        for k := k0; k < kEnd; k++ {
                            aTemp := A[i][k]
                            for j := j0; j < jEnd; j++ {
                                C[i][j] += aTemp * B[k][j]
                            }
                        }
                    }
                }
            }
        }()
    }

    // 分发工作
    for i := 0; i < n; i++ {
        jobs <- i
    }
    close(jobs)
    
    // 等待所有工作完成
    wg.Wait()

    return C
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func benchmark(size int) {
    // 预热
    runtime.GC()
    
    // 创建并初始化矩阵
    A := make([][]float64, size)
    B := make([][]float64, size)
    for i := range A {
        A[i] = make([]float64, size)
        B[i] = make([]float64, size)
        for j := range A[i] {
            A[i][j] = rand.Float64()
            B[i][j] = rand.Float64()
        }
    }

    // 执行并计时
    start := time.Now()
    C := matrixMultiply(A, B)
    duration := time.Since(start)
    timeMs := float64(duration.Nanoseconds()) / 1e6

    // 防止编译器优化
    sum := 0.0
    for i := range C {
        for j := range C[i] {
            sum += C[i][j]
        }
    }
    
    // 输出结果
    fmt.Printf("Go benchmark (size=%d): Time=%.2fms\n", size, timeMs)

    // 保存结果
    file, _ := os.OpenFile("../results/results.csv", os.O_APPEND|os.O_WRONLY, 0644)
    writer := csv.NewWriter(file)
    writer.Write([]string{
        "matrix_multiplication",
        "Go",
        strconv.Itoa(size),
        fmt.Sprintf("%.2f", timeMs),
    })
    writer.Flush()
    file.Close()
}

func main() {
    // 设置线程数
    runtime.GOMAXPROCS(runtime.NumCPU())
    
    sizes := []int{64, 128, 256, 512, 1024}
    for _, size := range sizes {
        benchmark(size)
    }
} 