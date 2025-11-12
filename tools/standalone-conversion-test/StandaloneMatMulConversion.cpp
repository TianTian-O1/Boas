//===- StandaloneMatMulConversion.cpp - Standalone conversion test -------===//
//
// This file demonstrates the MatMul conversion logic in a standalone program
// that doesn't require the Boas Dialect to be fully compiled.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

/// Helper function that demonstrates MatMul conversion logic
/// This is the same logic from BoasToLinalg.cpp
Value createLinalgMatMul(OpBuilder &builder, Location loc,
                          Value lhs, Value rhs,
                          RankedTensorType resultType) {
  // Step 1: Create empty output tensor
  Value emptyTensor = builder.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());

  // Step 2: Create zero constant
  Value zero;
  Type elemType = resultType.getElementType();
  if (elemType.isF32()) {
    zero = builder.create<arith::ConstantOp>(
        loc, builder.getF32FloatAttr(0.0));
  } else if (elemType.isF64()) {
    zero = builder.create<arith::ConstantOp>(
        loc, builder.getF64FloatAttr(0.0));
  } else {
    llvm::errs() << "Unsupported element type\n";
    return nullptr;
  }

  // Step 3: Fill output tensor with zeros
  Value initTensor = builder.create<linalg::FillOp>(
      loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

  // Step 4: Create linalg.matmul operation
  Value result = builder.create<linalg::MatmulOp>(
      loc, resultType, ValueRange{lhs, rhs}, ValueRange{initTensor})
      .getResult(0);

  return result;
}

/// Creates a test function showing MatMul conversion
func::FuncOp createTestFunction(MLIRContext *context, OpBuilder &builder) {
  Location loc = builder.getUnknownLoc();

  // Create function signature: (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  auto f32Type = builder.getF32Type();
  auto lhsType = RankedTensorType::get({2, 3}, f32Type);
  auto rhsType = RankedTensorType::get({3, 4}, f32Type);
  auto resultType = RankedTensorType::get({2, 4}, f32Type);

  auto funcType = builder.getFunctionType({lhsType, rhsType}, {resultType});

  // Create function
  auto funcOp = builder.create<func::FuncOp>(
      loc, "matmul_2x3_3x4", funcType);

  // Create function body
  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Value lhs = entryBlock->getArgument(0);
  Value rhs = entryBlock->getArgument(1);

  // Use our conversion helper
  Value result = createLinalgMatMul(builder, loc, lhs, rhs, resultType);

  // Return result
  builder.create<func::ReturnOp>(loc, ValueRange{result});

  return funcOp;
}

int main(int argc, char **argv) {
  // Initialize MLIR
  MLIRContext context;
  context.loadDialect<func::FuncDialect,
                       tensor::TensorDialect,
                       linalg::LinalgDialect,
                       arith::ArithDialect>();

  OpBuilder builder(&context);

  // Create module
  Location loc = builder.getUnknownLoc();
  auto module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Create test function
  auto funcOp = createTestFunction(&context, builder);

  // Print the generated IR
  llvm::outs() << "// Generated MatMul Conversion:\n";
  llvm::outs() << "// This demonstrates the conversion from Boas MatMulOp\n";
  llvm::outs() << "// to Linalg matmul with proper initialization.\n\n";

  module->print(llvm::outs());
  llvm::outs() << "\n";

  llvm::outs() << "\n// Conversion steps:\n";
  llvm::outs() << "// 1. tensor.empty() - Create output tensor\n";
  llvm::outs() << "// 2. arith.constant 0.0 - Create zero constant\n";
  llvm::outs() << "// 3. linalg.fill - Initialize output with zeros\n";
  llvm::outs() << "// 4. linalg.matmul - Perform matrix multiplication\n";

  return 0;
}
