//===- boas-opt.cpp - Boas optimizer driver ----------------------------===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Boas/Conversion/BoasToLinalg/BoasToLinalg.h"

// Forward declaration for pass registration
namespace mlir {
namespace boas {
void registerBoasConversionPasses() {
  // Register the Boas to Linalg conversion pass
  PassRegistration<ConvertBoasToLinalgPass>();
}
} // namespace boas
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  // Register Boas-specific passes
  mlir::boas::registerBoasConversionPasses();

  mlir::DialectRegistry registry;
  // Register standard MLIR dialects
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Boas optimizer driver\n", registry));
}
