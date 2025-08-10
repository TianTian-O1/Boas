//===----------------------------------------------------------------------===//
// Boas NPU Optimization Pass Implementation
//===----------------------------------------------------------------------===//

#include "mlirops/BoasDialect/BoasPasses.h"
#include "mlirops/BoasDialect/BoasDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::boas;

//===----------------------------------------------------------------------===//
// NPU Optimization Utilities
//===----------------------------------------------------------------------===//

namespace {

/// NPU优化配置
struct NPUConfig {
  int64_t blockM = 128;
  int64_t blockN = 256;
  int64_t blockK = 256;
  bool enableDiagonalTiling = true;
  StringRef strategy = "auto";
  
  /// 根据矩阵大小自动选择配置
  static NPUConfig getOptimalConfig(int64_t M, int64_t N, int64_t K) {
    NPUConfig config;
    
    // 计算块数量
    int64_t numBlocksM = (M + config.blockM - 1) / config.blockM;
    int64_t numBlocksN = (N + config.blockN - 1) / config.blockN;
    
    // 对角线分核策略选择
    config.enableDiagonalTiling = (numBlocksM >= 8 && numBlocksN >= 8);
    config.strategy = config.enableDiagonalTiling ? "diagonal" : "sequential";
    
    // 针对小矩阵调整块大小
    if (M < 512 || N < 512) {
      config.blockM = 64;
      config.blockN = 128;
      config.blockK = 128;
    }
    
    return config;
  }
};

/// 分析矩阵形状并提供优化建议
class MatrixShapeAnalysis {
public:
  static std::optional<std::pair<int64_t, int64_t>> 
  getStaticShape(TensorType tensorType) {
    ArrayRef<int64_t> shape = tensorType.getShape();
    if (shape.size() >= 2 && shape[shape.size()-2] >= 0 && shape[shape.size()-1] >= 0) {
      return std::make_pair(shape[shape.size()-2], shape[shape.size()-1]);
    }
    return std::nullopt;
  }
  
  static bool isLargeMatrix(int64_t M, int64_t N, int64_t K) {
    return (M * N * K) > (512 * 512 * 512);  // 128M elements threshold
  }
  
  static bool benefitsFromDiagonalTiling(int64_t M, int64_t N, int64_t K, 
                                        int64_t blockM, int64_t blockN) {
    int64_t numBlocksM = (M + blockM - 1) / blockM;
    int64_t numBlocksN = (N + blockN - 1) / blockN;
    return numBlocksM >= 8 && numBlocksN >= 8;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// NPU Optimization Patterns
//===----------------------------------------------------------------------===//

namespace {

/// 为矩阵乘法添加NPU优化配置的模式
struct AddNPUOptimizationToMatmul : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    // 如果已经有NPU优化配置，跳过
    if (op.hasNPUOptimization()) {
      return failure();
    }
    
    // 检查操作数是否在NPU设备上
    auto lhsType = op.getLhs().getType().cast<TensorType>();
    auto rhsType = op.getRhs().getType().cast<TensorType>();
    
    if (lhsType.getDevice() != "npu" && rhsType.getDevice() != "npu") {
      return failure();  // 不在NPU设备上，不需要NPU优化
    }
    
    // 分析矩阵形状
    auto lhsShape = MatrixShapeAnalysis::getStaticShape(lhsType);
    auto rhsShape = MatrixShapeAnalysis::getStaticShape(rhsType);
    
    if (!lhsShape || !rhsShape) {
      // 动态形状，使用默认配置
      return addDefaultNPUOptimization(op, rewriter);
    }
    
    // 静态形状，计算最优配置
    int64_t M = lhsShape->first;
    int64_t K = lhsShape->second;
    int64_t N = rhsShape->second;
    
    if (K != rhsShape->first) {
      return failure();  // 形状不匹配
    }
    
    auto config = NPUConfig::getOptimalConfig(M, N, K);
    return addOptimalNPUOptimization(op, rewriter, config);
  }

private:
  LogicalResult addDefaultNPUOptimization(MatmulOp op, PatternRewriter &rewriter) const {
    auto context = op.getContext();
    
    auto blockM = IntegerAttr::get(rewriter.getI64Type(), 128);
    auto blockN = IntegerAttr::get(rewriter.getI64Type(), 256); 
    auto blockK = IntegerAttr::get(rewriter.getI64Type(), 256);
    auto useDiagonalTiling = BoolAttr::get(context, true);
    auto strategy = StringAttr::get(context, "auto");
    
    auto npuOpt = NPUOptimizationAttr::get(context, blockM, blockN, blockK,
                                          useDiagonalTiling, strategy);
    
    auto newOp = rewriter.create<MatmulOp>(
        op.getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs(), npuOpt);
    
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
  
  LogicalResult addOptimalNPUOptimization(MatmulOp op, PatternRewriter &rewriter,
                                         const NPUConfig &config) const {
    auto context = op.getContext();
    
    auto blockM = IntegerAttr::get(rewriter.getI64Type(), config.blockM);
    auto blockN = IntegerAttr::get(rewriter.getI64Type(), config.blockN);
    auto blockK = IntegerAttr::get(rewriter.getI64Type(), config.blockK);
    auto useDiagonalTiling = BoolAttr::get(context, config.enableDiagonalTiling);
    auto strategy = StringAttr::get(context, config.strategy);
    
    auto npuOpt = NPUOptimizationAttr::get(context, blockM, blockN, blockK,
                                          useDiagonalTiling, strategy);
    
    auto newOp = rewriter.create<MatmulOp>(
        op.getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs(), npuOpt);
    
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// 优化NPU内存访问模式的模式
struct OptimizeNPUMemoryAccess : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    if (!op.hasNPUOptimization()) {
      return failure();
    }
    
    auto npuOpt = op.getNpuOpt().value();
    
    // 检查是否需要内存布局优化
    auto lhsType = op.getLhs().getType().cast<TensorType>();
    auto rhsType = op.getRhs().getType().cast<TensorType>();
    
    // 分析内存访问模式
    bool needsOptimization = false;
    
    // 检查是否需要数据重排以优化内存访问
    if (shouldOptimizeDataLayout(lhsType, rhsType, npuOpt)) {
      needsOptimization = true;
    }
    
    if (!needsOptimization) {
      return failure();
    }
    
    // 创建优化的操作
    return optimizeMatmulMemoryLayout(op, rewriter);
  }

private:
  bool shouldOptimizeDataLayout(TensorType lhsType, TensorType rhsType,
                               NPUOptimizationAttr npuOpt) const {
    // 检查块大小是否与内存对齐要求匹配
    int64_t blockM = npuOpt.getBlockM().getInt();
    int64_t blockN = npuOpt.getBlockN().getInt();
    
    // NPU更喜欢512B对齐的访问
    return (blockM * sizeof(float)) % 512 != 0 || 
           (blockN * sizeof(float)) % 512 != 0;
  }
  
  LogicalResult optimizeMatmulMemoryLayout(MatmulOp op, PatternRewriter &rewriter) const {
    // 在实际实现中，这里会插入内存重排操作
    // 目前只是添加优化提示属性
    
    auto newOp = rewriter.create<MatmulOp>(
        op.getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs(), op.getNpuOpt());
    
    // 添加内存优化提示
    newOp->setAttr("boas.memory_optimized", rewriter.getBoolAttr(true));
    
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// 自动设备选择模式
struct AutoDeviceSelection : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType().cast<TensorType>();
    auto rhsType = op.getRhs().getType().cast<TensorType>();
    
    // 如果操作数已经在NPU上，不需要修改
    if (lhsType.getDevice() == "npu" && rhsType.getDevice() == "npu") {
      return failure();
    }
    
    // 分析是否适合NPU执行
    if (!shouldUseNPU(lhsType, rhsType)) {
      return failure();
    }
    
    // 创建设备转移操作
    return moveToNPU(op, rewriter);
  }

private:
  bool shouldUseNPU(TensorType lhsType, TensorType rhsType) const {
    // 检查矩阵大小是否适合NPU
    auto lhsShape = MatrixShapeAnalysis::getStaticShape(lhsType);
    auto rhsShape = MatrixShapeAnalysis::getStaticShape(rhsType);
    
    if (!lhsShape || !rhsShape) {
      return true;  // 动态形状，假设适合NPU
    }
    
    int64_t M = lhsShape->first;
    int64_t K = lhsShape->second;
    int64_t N = rhsShape->second;
    
    // 大矩阵更适合NPU
    return MatrixShapeAnalysis::isLargeMatrix(M, N, K);
  }
  
  LogicalResult moveToNPU(MatmulOp op, PatternRewriter &rewriter) const {
    auto context = op.getContext();
    
    // 创建到NPU的设备转移操作
    auto lhsType = op.getLhs().getType().cast<TensorType>();
    auto rhsType = op.getRhs().getType().cast<TensorType>();
    
    auto npuLhsType = TensorType::get(context, lhsType.getShape(),
                                     lhsType.getElementType(), "npu");
    auto npuRhsType = TensorType::get(context, rhsType.getShape(),
                                     rhsType.getElementType(), "npu");
    
    Value npuLhs = rewriter.create<ToDeviceOp>(
        op.getLoc(), npuLhsType, op.getLhs(), rewriter.getStringAttr("npu"));
    Value npuRhs = rewriter.create<ToDeviceOp>(
        op.getLoc(), npuRhsType, op.getRhs(), rewriter.getStringAttr("npu"));
    
    // 创建NPU上的矩阵乘法
    auto npuResultType = TensorType::get(context, 
        op.getResult().getType().cast<TensorType>().getShape(),
        op.getResult().getType().cast<TensorType>().getElementType(), "npu");
    
    auto npuMatmul = rewriter.create<MatmulOp>(
        op.getLoc(), npuResultType, npuLhs, npuRhs, std::nullopt);
    
    rewriter.replaceOp(op, npuMatmul.getResult());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct BoasNPUOptimizationPass : public PassWrapper<BoasNPUOptimizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BoasNPUOptimizationPass)

  BoasNPUOptimizationPass() = default;
  BoasNPUOptimizationPass(const BoasNPUOptimizationPass& other) : PassWrapper(other) {
    targetDevice = other.targetDevice;
    enableDiagonalTiling = other.enableDiagonalTiling;
    blockSizeM = other.blockSizeM;
    blockSizeN = other.blockSizeN;
    blockSizeK = other.blockSizeK;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BoasDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto context = &getContext();

    // 设置优化模式
    RewritePatternSet patterns(context);
    patterns.add<AddNPUOptimizationToMatmul, OptimizeNPUMemoryAccess, AutoDeviceSelection>(context);

    // 应用优化
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "boas-npu-opt"; }
  StringRef getDescription() const final {
    return "NPU-specific optimizations for Boas operations";
  }

private:
  // Pass选项
  std::string targetDevice = "npu";
  bool enableDiagonalTiling = true;
  int64_t blockSizeM = 128;
  int64_t blockSizeN = 256;
  int64_t blockSizeK = 256;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::boas::createBoasNPUOptimizationPass() {
  return std::make_unique<BoasNPUOptimizationPass>();
}
