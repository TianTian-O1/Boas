#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

// 基础矩阵乘法CUDA kernel
// C = A * B，其中 A: M x K, B: K x N, C: M x N
__global__ void matmul_kernel_basic(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 使用共享内存优化的矩阵乘法kernel（分块大小32x32）
template<int BLOCK_SIZE = 32>
__global__ void matmul_kernel_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    // 共享内存用于tile缓存
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 全局位置
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // 对K维度分块
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 加载tile到共享内存
        int aCol = t * BLOCK_SIZE + tx;
        int bRow = t * BLOCK_SIZE + ty;

        if (row < M && aCol < K)
            As[ty][tx] = A[row * K + aCol];
        else
            As[ty][tx] = 0.0f;

        if (bRow < K && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // 计算部分点积
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 针对大矩阵优化的kernel（分块大小64x64，每个线程计算4x4子块）
__global__ void matmul_kernel_large(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    const int BLOCK_SIZE = 64;
    const int THREAD_TILE = 4;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 每个线程负责THREAD_TILE x THREAD_TILE的输出
    int rowBase = by * BLOCK_SIZE + ty * THREAD_TILE;
    int colBase = bx * BLOCK_SIZE + tx * THREAD_TILE;

    float sum[THREAD_TILE][THREAD_TILE] = {0};

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 协作加载tile
        for (int i = 0; i < THREAD_TILE; i++) {
            int row = rowBase + i;
            int col = t * BLOCK_SIZE + tx * THREAD_TILE;
            if (row < M && col < K)
                As[ty * THREAD_TILE + i][tx * THREAD_TILE] = A[row * K + col];
        }

        for (int i = 0; i < THREAD_TILE; i++) {
            int row = t * BLOCK_SIZE + ty * THREAD_TILE;
            int col = colBase + i;
            if (row < K && col < N)
                Bs[ty * THREAD_TILE][tx * THREAD_TILE + i] = B[row * N + col];
        }

        __syncthreads();

        // 计算
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++) {
                    sum[i][j] += As[ty * THREAD_TILE + i][k] * Bs[k][tx * THREAD_TILE + j];
                }
            }
        }

        __syncthreads();
    }

    // 写入结果
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = rowBase + i;
            int col = colBase + j;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}

// C++封装函数：根据矩阵大小选择最优kernel
void launch_matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0) {

    // 根据矩阵大小选择策略
    if (M < 512 && N < 512 && K < 512) {
        // 小矩阵：使用基础kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (M + 15) / 16);
        matmul_kernel_basic<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
    }
    else if (M >= 2048 && N >= 2048 && K >= 2048) {
        // 大矩阵：使用高级优化kernel
        dim3 blockDim(16, 16);  // 64x64 block with 4x4 per thread
        dim3 gridDim((N + 63) / 64, (M + 63) / 64);
        matmul_kernel_large<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
    }
    else {
        // 中等矩阵：使用共享内存优化kernel
        dim3 blockDim(32, 32);
        dim3 gridDim((N + 31) / 32, (M + 31) / 32);
        matmul_kernel_shared<32><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
    }
}

} // extern "C"
