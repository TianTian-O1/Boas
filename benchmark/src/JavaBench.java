import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class JavaBench {
    private static final int BLOCK_SIZE = 32;
    private static final ExecutorService executor = 
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    private static class MatrixMultTask implements Runnable {
        private final double[][] A, B, C;
        private final int i0, iEnd, n;

        public MatrixMultTask(double[][] A, double[][] B, double[][] C, 
                            int i0, int iEnd) {
            this.A = A;
            this.B = B;
            this.C = C;
            this.i0 = i0;
            this.iEnd = iEnd;
            this.n = A.length;
        }

        @Override
        public void run() {
            for (int j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
                int jEnd = Math.min(j0 + BLOCK_SIZE, n);
                for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                    int kEnd = Math.min(k0 + BLOCK_SIZE, n);
                    
                    // 优化的块计算
                    for (int i = i0; i < iEnd; i++) {
                        for (int k = k0; k < kEnd; k++) {
                            double aTemp = A[i][k];
                            for (int j = j0; j < jEnd; j++) {
                                C[i][j] += aTemp * B[k][j];
                            }
                        }
                    }
                }
            }
        }
    }

    private static void matrixMultiply(double[][] A, double[][] B, double[][] C) 
            throws InterruptedException, ExecutionException {
        int n = A.length;
        List<Future<?>> futures = new ArrayList<>();

        // 并行计算各个块
        for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
            int iEnd = Math.min(i0 + BLOCK_SIZE, n);
            futures.add(executor.submit(
                new MatrixMultTask(A, B, C, i0, iEnd)
            ));
        }

        // 等待所有任务完成
        for (Future<?> future : futures) {
            future.get();
        }
    }

    private static void benchmark(int size) {
        try {
            // 预热 JVM
            System.gc();
            Thread.sleep(100);

            // 初始化矩阵
            Random rand = new Random();
            double[][] A = new double[size][size];
            double[][] B = new double[size][size];
            double[][] C = new double[size][size];

            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    A[i][j] = rand.nextDouble();
                    B[i][j] = rand.nextDouble();
                }
            }

            // 执行并计时
            long startTime = System.nanoTime();
            matrixMultiply(A, B, C);
            long endTime = System.nanoTime();
            double timeMs = (endTime - startTime) / 1_000_000.0;

            // 防止编译器优化
            double sum = 0;
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    sum += C[i][j];
                }
            }

            // 输出结果
            System.out.printf("Java benchmark (size=%d): Time=%.2fms%n", 
                            size, timeMs);

            // 保存结果
            try (FileWriter fw = new FileWriter("../results/results.csv", true);
                 BufferedWriter bw = new BufferedWriter(fw)) {
                bw.write(String.format("matrix_multiplication,Java,%d,%.2f%n", 
                        size, timeMs));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            int[] sizes = {64, 128, 256, 512, 1024};
            for (int size : sizes) {
                benchmark(size);
            }
        } finally {
            executor.shutdown();
        }
    }
} 