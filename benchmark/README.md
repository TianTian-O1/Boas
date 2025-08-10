# 🚀 Boas语言NPU性能Benchmark

## 📊 **性能对比基准测试**

本benchmark对比以下几种实现的矩阵乘法性能：

1. **🎯 Boas语言 + CANN NPU** - 我们的实现
2. **🔥 PyTorch + torch_npu** - 官方NPU实现  
3. **⚡ Triton-Ascend** - 参考实现
4. **💻 CPU baseline** - NumPy/PyTorch CPU

## 🏗️ **测试矩阵规模**

| 规模类别 | 矩阵大小 | 内存使用 | 计算量 |
|---------|---------|---------|--------|
| 小规模 | 64×64 | ~32KB | 262K FLOP |
| 中规模 | 256×256 | ~512KB | 33M FLOP |
| 大规模 | 1024×1024 | ~8MB | 2.1B FLOP |
| 超大规模 | 2048×2048 | ~32MB | 17B FLOP |
| 极大规模 | 4096×4096 | ~128MB | 137B FLOP |

## 📈 **性能指标**

- **执行时间** (ms)
- **吞吐量** (GFLOPS)
- **内存带宽** (GB/s) 
- **NPU利用率** (%)
- **编译时间** (s)

## 🔧 **测试环境**

- **硬件**: 昇腾910B2 NPU
- **软件**: CANN 8.1.RC1
- **对比框架**: PyTorch 2.0 + torch_npu
- **Triton**: triton-ascend latest
