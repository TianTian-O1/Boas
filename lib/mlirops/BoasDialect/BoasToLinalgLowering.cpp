//===----------------------------------------------------------------------===//
// Boas to Linalg Lowering Pass Implementation
//===----------------------------------------------------------------------===//

#include "mlirops/BoasDialect/BoasPasses.h"
#include "mlirops/BoasDialect/BoasDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::boas;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// 将boas.matmul转换为linalg.matmul的模式
struct MatmulOpLowering : public OpConversionPattern<MatmulOp> {
  using OpConversionPattern<MatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    // 获取操作数
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    
    // 转换Boas tensor类型到标准tensor类型
    auto lhsTensorType = convertBoasToTensorType(op.getLhs().getType().cast<TensorType>());
    auto rhsTensorType = convertBoasToTensorType(op.getRhs().getType().cast<TensorType>());
    auto resultTensorType = convertBoasToTensorType(op.getResult().getType().cast<TensorType>());
    
    // 创建输出张量
    Value outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), resultTensorType.getElementType());
    
    // 初始化输出张量为零
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultTensorType.getElementType()));
    Value initializedOutput = rewriter.create<linalg::FillOp>(
        loc, zero, outputTensor).getResult(0);
    
    // 检查是否有NPU优化配置
    if (op.hasNPUOptimization()) {
      // 对于NPU优化的情况，创建带优化属性的linalg.matmul
      auto npuOpt = op.getNpuOpt().value();
      
      // 创建包含NPU优化信息的属性
      SmallVector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("boas.npu_optimized", rewriter.getBoolAttr(true)));
      attrs.push_back(rewriter.getNamedAttr("boas.block_m", npuOpt.getBlockM()));
      attrs.push_back(rewriter.getNamedAttr("boas.block_n", npuOpt.getBlockN()));
      attrs.push_back(rewriter.getNamedAttr("boas.block_k", npuOpt.getBlockK()));
      attrs.push_back(rewriter.getNamedAttr("boas.diagonal_tiling", npuOpt.getUseDiagonalTiling()));
      attrs.push_back(rewriter.getNamedAttr("boas.strategy", npuOpt.getStrategy()));
      
      // 创建带优化属性的matmul
      auto matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, TypeRange{resultTensorType}, ValueRange{lhs, rhs}, ValueRange{initializedOutput});
      
      // 添加优化属性
      for (auto attr : attrs) {
        matmulOp->setAttr(attr.getName(), attr.getValue());
      }
      
      rewriter.replaceOp(op, matmulOp.getResult(0));
    } else {
      // 标准linalg.matmul
      auto matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, TypeRange{resultTensorType}, ValueRange{lhs, rhs}, ValueRange{initializedOutput});
      
      rewriter.replaceOp(op, matmulOp.getResult(0));
    }
    
    return success();
  }

private:
  /// 将Boas tensor类型转换为标准tensor类型
  RankedTensorType convertBoasToTensorType(TensorType boasType) const {
    return RankedTensorType::get(boasType.getShape(), boasType.getElementType());
  }
};

/// 将boas.tensor.create转换为tensor.empty + tensor.insert的模式
struct TensorCreateOpLowering : public OpConversionPattern<TensorCreateOp> {
  using OpConversionPattern<TensorCreateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TensorCreateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    // 获取维度
    Value rows = adaptor.getRows();
    Value cols = adaptor.getCols();
    
    // 创建动态形状的tensor
    auto resultType = convertBoasToTensorType(op.getResult().getType().cast<TensorType>());
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(), ValueRange{rows, cols});
    
    // 如果有初始值，需要将其填充到tensor中
    Value values = adaptor.getValues();
    if (auto memrefType = values.getType().dyn_cast<MemRefType>()) {
      // 从memref中加载值并填充到tensor
      Value result = fillTensorFromMemRef(rewriter, loc, emptyTensor, values, rows, cols);
      rewriter.replaceOp(op, result);
    } else {
      // 直接使用空tensor
      rewriter.replaceOp(op, emptyTensor);
    }
    
    return success();
  }

private:
  RankedTensorType convertBoasToTensorType(TensorType boasType) const {
    return RankedTensorType::get(boasType.getShape(), boasType.getElementType());
  }
  
  Value fillTensorFromMemRef(ConversionPatternRewriter &rewriter, Location loc,
                            Value tensor, Value memref, Value rows, Value cols) const {
    // 创建循环来填充tensor
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    auto forOpI = rewriter.create<scf::ForOp>(loc, c0, rows, c1, ValueRange{tensor});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOpI.getBody());
      
      Value i = forOpI.getInductionVar();
      Value tensorI = forOpI.getRegionIterArg(0);
      
      auto forOpJ = rewriter.create<scf::ForOp>(loc, c0, cols, c1, ValueRange{tensorI});
      {
        rewriter.setInsertionPointToStart(forOpJ.getBody());
        
        Value j = forOpJ.getInductionVar();
        Value tensorIJ = forOpJ.getRegionIterArg(0);
        
        // 从memref加载值
        Value val = rewriter.create<memref::LoadOp>(loc, memref, ValueRange{i, j});
        
        // 插入到tensor
        Value newTensor = rewriter.create<tensor::InsertOp>(loc, val, tensorIJ, ValueRange{i, j});
        
        rewriter.create<scf::YieldOp>(loc, newTensor);
      }
      
      rewriter.create<scf::YieldOp>(loc, forOpJ.getResult(0));
    }
    
    return forOpI.getResult(0);
  }
};

/// 将boas.tensor.random转换为运行时调用的模式
struct TensorRandomOpLowering : public OpConversionPattern<TensorRandomOp> {
  using OpConversionPattern<TensorRandomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TensorRandomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    // 创建空tensor
    auto resultType = convertBoasToTensorType(op.getResult().getType().cast<TensorType>());
    Value rows = adaptor.getRows();
    Value cols = adaptor.getCols();
    
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(), ValueRange{rows, cols});
    
    // 创建运行时函数调用来填充随机值
    auto moduleOp = op->getParentOfType<ModuleOp>();
    func::FuncOp randomFunc = getOrCreateRandomFunction(rewriter, moduleOp, resultType.getElementType());
    
    // 调用随机填充函数
    auto callOp = rewriter.create<func::CallOp>(
        loc, randomFunc, ValueRange{emptyTensor, rows, cols});
    
    rewriter.replaceOp(op, callOp.getResult(0));
    return success();
  }

private:
  RankedTensorType convertBoasToTensorType(TensorType boasType) const {
    return RankedTensorType::get(boasType.getShape(), boasType.getElementType());
  }
  
  func::FuncOp getOrCreateRandomFunction(ConversionPatternRewriter &rewriter, 
                                        ModuleOp module, Type elementType) const {
    StringRef funcName = "boas_tensor_random_fill";
    
    if (auto existingFunc = module.lookupSymbol<func::FuncOp>(funcName)) {
      return existingFunc;
    }
    
    // 创建随机填充函数声明
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    
    auto tensorType = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
    auto indexType = rewriter.getIndexType();
    
    auto funcType = rewriter.getFunctionType(
        {tensorType, indexType, indexType}, {tensorType});
    
    auto funcOp = rewriter.create<func::FuncOp>(
        module.getLoc(), funcName, funcType);
    funcOp.setPrivate();
    
    return funcOp;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct BoasToLinalgLoweringPass : public PassWrapper<BoasToLinalgLoweringPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BoasToLinalgLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, 
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto context = &getContext();

    // 设置类型转换器
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    
    // Boas tensor类型转换为标准tensor类型
    typeConverter.addConversion([context](TensorType boasType) -> Type {
      return RankedTensorType::get(boasType.getShape(), boasType.getElementType());
    });

    // 设置转换目标
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, 
                          arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    target.addIllegalDialect<BoasDialect>();

    // 设置转换模式
    RewritePatternSet patterns(context);
    patterns.add<MatmulOpLowering, TensorCreateOpLowering, TensorRandomOpLowering>(
        typeConverter, context);

    // 执行转换
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "boas-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower Boas dialect to Linalg dialect";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::boas::createBoasToLinalgLoweringPass() {
  return std::make_unique<BoasToLinalgLoweringPass>();
}
