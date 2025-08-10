//===----------------------------------------------------------------------===//
// Boas Dialect Integration with Existing MLIRGen
//===----------------------------------------------------------------------===//

#include "mlirops/BoasDialect/BoasDialect.h"
#include "mlirops/BoasDialect/BoasPasses.h"
#include "mlirops/MLIRGen.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"

using namespace mlir;
using namespace mlir::boas;

namespace matrix {

//===----------------------------------------------------------------------===//
// Boas Dialect Integration
//===----------------------------------------------------------------------===//

/// 集成Boas dialect到现有的MLIRGen框架
class BoasMLIRGenIntegration {
public:
  /// 初始化Boas dialect支持
  static void initializeBoasDialect(MLIRContext* context) {
    // 注册Boas dialect
    context->loadDialect<BoasDialect>();
    
    // 注册Boas passes
    registerBoasPasses();
  }
  
  /// 创建Boas优化管道
  static void buildBoasOptimizationPipeline(PassManager& pm) {
    // 1. Boas dialect内的高级优化
    pm.addPass(createBoasMatrixOptimizationPass());
    pm.addPass(createBoasNPUOptimizationPass());
    pm.addPass(createDeviceAwareOptimizationPass());
    
    // 2. Lowering到标准dialect
    pm.addPass(createBoasToLinalgLoweringPass());
    
    // 3. 标准优化passes
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    // 4. Linalg优化
    pm.addPass(linalg::createLinalgTilingPass());
    pm.addPass(linalg::createLinalgVectorizationPass());
    
    // 5. NPU特定的kernel生成
    pm.addPass(createNPUKernelGenerationPass());
  }
};

//===----------------------------------------------------------------------===//
// Extended MLIRGen with Boas Dialect Support
//===----------------------------------------------------------------------===//

/// 扩展的MLIRGen类，支持Boas dialect
class BoasMLIRGen : public MLIRGen {
public:
  BoasMLIRGen() : MLIRGen() {
    // 初始化Boas dialect支持
    BoasMLIRGenIntegration::initializeBoasDialect(getContext());
  }
  
  /// 使用Boas dialect生成矩阵乘法
  mlir::Value generateBoasMatmul(const MatmulExprAST* expr) {
    auto loc = getBuilder()->getUnknownLoc();
    
    // 生成操作数
    auto lhs = generateMLIRForNode(expr->getLHS());
    auto rhs = generateMLIRForNode(expr->getRHS());
    
    if (!lhs || !rhs) {
      std::cerr << "Failed to generate operands for Boas matmul\n";
      return nullptr;
    }
    
    // 转换为Boas tensor类型
    auto lhsTensorType = convertToBoasTensorType(lhs);
    auto rhsTensorType = convertToBoasTensorType(rhs);
    
    if (!lhsTensorType || !rhsTensorType) {
      std::cerr << "Failed to convert to Boas tensor types\n";
      return nullptr;
    }
    
    // 推断结果类型
    auto resultType = inferBoasMatmulResultType(lhsTensorType, rhsTensorType);
    
    // 创建Boas matmul操作
    auto matmulOp = getBuilder()->create<MatmulOp>(
        loc, resultType, lhs, rhs, std::nullopt);
    
    return matmulOp.getResult();
  }
  
  /// 生成Boas tensor创建操作
  mlir::Value generateBoasTensorCreate(const TensorCreateExprAST* expr) {
    auto loc = getBuilder()->getUnknownLoc();
    
    // 获取维度参数
    auto rows = generateMLIRForNode(expr->getRows());
    auto cols = generateMLIRForNode(expr->getCols());
    auto values = generateMLIRForNode(expr->getValues());
    
    if (!rows || !cols || !values) {
      return nullptr;
    }
    
    // 确定设备（默认CPU，可根据需要选择NPU）
    StringRef device = shouldUseNPU(expr) ? "npu" : "cpu";
    
    // 推断元素类型
    Type elementType = inferElementType(values);
    auto resultType = TensorType::get(getContext(), {-1, -1}, elementType, device);
    
    // 创建Boas tensor.create操作
    auto createOp = getBuilder()->create<TensorCreateOp>(
        loc, resultType, rows, cols, values, getBuilder()->getStringAttr(device));
    
    return createOp.getResult();
  }
  
  /// 生成Boas tensor随机创建操作
  mlir::Value generateBoasTensorRandom(const TensorRandomExprAST* expr) {
    auto loc = getBuilder()->getUnknownLoc();
    
    auto rows = generateMLIRForNode(expr->getRows());
    auto cols = generateMLIRForNode(expr->getCols());
    
    if (!rows || !cols) {
      return nullptr;
    }
    
    // 选择设备
    StringRef device = shouldUseNPU(expr) ? "npu" : "cpu";
    
    // 使用f32作为默认类型
    Type elementType = getBuilder()->getF32Type();
    auto resultType = TensorType::get(getContext(), {-1, -1}, elementType, device);
    
    // 创建Boas tensor.random操作
    auto randomOp = getBuilder()->create<TensorRandomOp>(
        loc, resultType, rows, cols, getBuilder()->getStringAttr(device), std::nullopt);
    
    return randomOp.getResult();
  }
  
  /// 运行Boas优化管道
  void runBoasOptimizations(ModuleOp module) {
    PassManager pm(getContext());
    BoasMLIRGenIntegration::buildBoasOptimizationPipeline(pm);
    
    if (failed(pm.run(module))) {
      std::cerr << "Failed to run Boas optimization pipeline\n";
    }
  }

private:
  /// 转换为Boas tensor类型
  TensorType convertToBoasTensorType(Value value) {
    auto memrefType = value.getType().dyn_cast<MemRefType>();
    if (!memrefType) {
      return nullptr;
    }
    
    // 推断设备（基于值的生成方式或显式标记）
    StringRef device = inferDevice(value);
    
    return TensorType::get(getContext(), memrefType.getShape(),
                          memrefType.getElementType(), device);
  }
  
  /// 推断Boas matmul结果类型
  TensorType inferBoasMatmulResultType(TensorType lhsType, TensorType rhsType) {
    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();
    
    SmallVector<int64_t> resultShape;
    if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
      int64_t M = lhsShape[lhsShape.size() - 2];
      int64_t N = rhsShape[rhsShape.size() - 1];
      resultShape = {M, N};
    }
    
    // 使用与操作数相同的设备
    StringRef device = lhsType.getDevice();
    
    return TensorType::get(getContext(), resultShape,
                          lhsType.getElementType(), device);
  }
  
  /// 推断设备
  StringRef inferDevice(Value value) {
    // 简单的启发式：大张量使用NPU，小张量使用CPU
    if (auto memrefType = value.getType().dyn_cast<MemRefType>()) {
      if (memrefType.hasStaticShape()) {
        int64_t numElements = memrefType.getNumElements();
        return (numElements > 1024) ? "npu" : "cpu";
      }
    }
    return "cpu";  // 默认CPU
  }
  
  /// 判断是否应该使用NPU
  bool shouldUseNPU(const ExprAST* expr) {
    // 简单策略：大操作使用NPU
    // 在实际实现中可以基于成本模型或用户注解
    return true;  // 暂时默认使用NPU
  }
  
  /// 推断元素类型
  Type inferElementType(Value value) {
    if (auto shapedType = value.getType().dyn_cast<ShapedType>()) {
      return shapedType.getElementType();
    }
    return getBuilder()->getF64Type();  // 默认f64
  }
  
  /// 获取context
  MLIRContext* getContext() {
    return getBuilder()->getContext();
  }
};

//===----------------------------------------------------------------------===//
// Factory Functions
//===----------------------------------------------------------------------===//

/// 创建支持Boas dialect的MLIRGen实例
std::unique_ptr<MLIRGen> createBoasMLIRGen() {
  return std::make_unique<BoasMLIRGen>();
}

/// 注册Boas dialect和passes
void registerBoasInfrastructure() {
  // 注册Boas dialect
  registerDialect<BoasDialect>();
  
  // 注册Boas passes
  registerBoasPasses();
}

} // namespace matrix

//===----------------------------------------------------------------------===//
// Pass Registration Implementation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace boas {

void registerBoasPasses() {
  PassRegistration<BoasToLinalgLoweringPass>();
  PassRegistration<BoasNPUOptimizationPass>();
  
  // 注册pass pipeline
  PassPipelineRegistration<>("boas-optimization-pipeline",
                            "Boas dialect optimization pipeline",
                            [](OpPassManager &pm) {
                              matrix::BoasMLIRGenIntegration::buildBoasOptimizationPipeline(pm);
                            });
}

} // namespace boas
} // namespace mlir
