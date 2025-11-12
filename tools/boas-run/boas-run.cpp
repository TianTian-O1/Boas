//===- boas-run.cpp - Boas Execution Tool -------------------------------===//
//
// This tool executes compiled Boas/Linalg programs using MLIR JIT execution.
// Usage:
//   boas-run input.mlir
//   boas-run input.mlir --entry-point=main
//   boas-run input.mlir -v
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// Command line options
static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-"));

static llvm::cl::opt<std::string> entryPoint("entry-point",
                                        llvm::cl::desc("Entry point function name"),
                                        llvm::cl::value_desc("function"),
                                        llvm::cl::init("main"));

static llvm::cl::opt<bool> verbose("v", llvm::cl::desc("Verbose output"), llvm::cl::init(false));

static llvm::cl::opt<bool> dumpIR("dump-ir",
                              llvm::cl::desc("Dump IR before execution"),
                              llvm::cl::init(false));

/// Load and parse the input file
static OwningOpRef<ModuleOp> loadModule(MLIRContext &context,
                                         llvm::StringRef filename) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Error opening input file: " << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

/// Configure the lowering pipeline to LLVM
static LogicalResult lowerToLLVM(ModuleOp module) {
  mlir::PassManager pm(module.getContext());

  if (verbose) {
    llvm::outs() << "Lowering pipeline:\n";
    llvm::outs() << "  1. Linalg → Loops\n";
    llvm::outs() << "  2. Bufferization\n";
    llvm::outs() << "  3. → LLVM IR\n";
  }

  // Stage 1: Lower Linalg to loops
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Stage 2: Bufferization (comprehensive)
  bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferDeallocationPass());

  // Stage 3: Lower to LLVM
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (verbose) {
    // Disable IR printing due to multithreading issues
    // pm.enableIRPrinting(...);
  }

  if (failed(pm.run(module))) {
    llvm::errs() << "Failed to lower to LLVM\n";
    return failure();
  }

  return success();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse command line options
  llvm::cl::ParseCommandLineOptions(argc, argv, "Boas Execution Engine\n");

  // Initialize LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Initialize MLIR context
  MLIRContext context;
  DialectRegistry registry;

  // Register all required dialects
  registry.insert<func::FuncDialect,
                  tensor::TensorDialect,
                  linalg::LinalgDialect,
                  arith::ArithDialect,
                  memref::MemRefDialect,
                  scf::SCFDialect,
                  bufferization::BufferizationDialect>();

  // Register LLVM IR translation
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Load input module
  if (verbose) {
    llvm::outs() << "Loading input file: " << inputFilename << "\n";
  }

  auto module = loadModule(context, inputFilename);
  if (!module) {
    llvm::errs() << "Failed to load input module\n";
    return 1;
  }

  if (dumpIR) {
    llvm::outs() << "// Input IR:\n";
    module->print(llvm::outs());
    llvm::outs() << "\n\n";
  }

  // Lower to LLVM
  if (verbose) {
    llvm::outs() << "Lowering to LLVM IR...\n";
  }

  if (failed(lowerToLLVM(*module))) {
    llvm::errs() << "Failed to lower to LLVM\n";
    return 1;
  }

  if (dumpIR) {
    llvm::outs() << "// Lowered IR:\n";
    module->print(llvm::outs());
    llvm::outs() << "\n\n";
  }

  // Create execution engine
  if (verbose) {
    llvm::outs() << "Creating execution engine...\n";
  }

  auto maybeEngine = ExecutionEngine::create(*module);
  if (!maybeEngine) {
    llvm::errs() << "Failed to create execution engine: "
                  << maybeEngine.takeError() << "\n";
    return 1;
  }

  auto &engine = maybeEngine.get();

  // Invoke the entry point
  if (verbose) {
    llvm::outs() << "Invoking entry point: " << entryPoint << "\n";
  }

  auto error = engine->invokePacked(entryPoint);
  if (error) {
    llvm::errs() << "Execution failed: " << error << "\n";
    return 1;
  }

  if (verbose) {
    llvm::outs() << "Execution successful!\n";
  }

  return 0;
}
