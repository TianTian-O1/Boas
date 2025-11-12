//===- boas-compile.cpp - Boas Compilation Tool -------------------------===//
//
// This tool compiles Boas programs through the full compilation pipeline.
// Usage:
//   boas-compile input.mlir -o output.mlir
//   boas-compile input.mlir --emit-linalg -o output.mlir
//   boas-compile input.mlir --emit-llvm -o output.ll
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

// Command line options
static cl::opt<std::string> inputFilename(cl::Positional,
                                           cl::desc("<input file>"),
                                           cl::init("-"));

static cl::opt<std::string> outputFilename("o",
                                            cl::desc("Output filename"),
                                            cl::value_desc("filename"),
                                            cl::init("-"));

static cl::opt<bool> emitLinalg("emit-linalg",
                                 cl::desc("Emit Linalg dialect IR"),
                                 cl::init(false));

static cl::opt<bool> emitLLVM("emit-llvm",
                               cl::desc("Emit LLVM IR"),
                               cl::init(false));

static cl::opt<bool> optimize("O",
                               cl::desc("Enable optimizations"),
                               cl::init(false));

static cl::opt<bool> verbose("v",
                              cl::desc("Verbose output"),
                              cl::init(false));

/// Load and parse the input file
static OwningOpRef<ModuleOp> loadModule(MLIRContext &context,
                                         StringRef filename) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Error opening input file: " << errorMessage << "\n";
    return nullptr;
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

/// Configure the compilation pipeline
static LogicalResult configurePipeline(PassManager &pm, bool emitLinalg,
                                        bool emitLLVM, bool optimize) {
  // Stage 1: Boas to Linalg lowering
  // Note: Commented out until Boas dialect is fully compiled
  // pm.addPass(mlir::boas::createConvertBoasToLinalgPass());

  if (verbose) {
    llvm::outs() << "Pipeline stages:\n";
    llvm::outs() << "  1. Boas → Linalg (when Boas dialect is ready)\n";
  }

  // Stage 2: Linalg optimizations (if requested)
  if (optimize) {
    // Add canonicalization and CSE
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (verbose) {
      llvm::outs() << "  2. Canonicalization and CSE\n";
    }
  }

  // Stage 3: Further lowering (if requested)
  if (emitLLVM) {
    if (verbose) {
      llvm::outs() << "  3. Linalg → LLVM (future work)\n";
    }
    // Future: Add Linalg to LLVM lowering passes
    // pm.addPass(createConvertLinalgToLoopsPass());
    // pm.addPass(createConvertSCFToCFPass());
    // pm.addPass(createConvertLinalgToLLVMPass());
  }

  return success();
}

/// Write the output module to file
static LogicalResult writeOutput(ModuleOp module, StringRef filename) {
  std::string errorMessage;
  auto outputFile = openOutputFile(filename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << "Error opening output file: " << errorMessage << "\n";
    return failure();
  }

  module->print(outputFile->os());
  outputFile->keep();
  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Parse command line options
  cl::ParseCommandLineOptions(argc, argv, "Boas Compiler\n");

  // Initialize MLIR context
  MLIRContext context;
  context.loadDialect<func::FuncDialect, tensor::TensorDialect,
                       linalg::LinalgDialect, arith::ArithDialect>();

  // Load input module
  if (verbose) {
    llvm::outs() << "Loading input file: " << inputFilename << "\n";
  }

  auto module = loadModule(context, inputFilename);
  if (!module) {
    llvm::errs() << "Failed to load input module\n";
    return 1;
  }

  // Configure and run compilation pipeline
  if (verbose) {
    llvm::outs() << "Configuring compilation pipeline...\n";
  }

  PassManager pm(&context);
  if (failed(configurePipeline(pm, emitLinalg, emitLLVM, optimize))) {
    llvm::errs() << "Failed to configure pipeline\n";
    return 1;
  }

  // Enable IR printing if verbose
  if (verbose) {
    pm.enableIRPrinting(
        [](Pass *, Operation *) { return false; },  // Before pass
        [](Pass *, Operation *) { return true; },   // After pass
        true,                                        // Print module scope
        true,                                        // Print after change
        false,                                       // Print after failure
        llvm::errs());
  }

  // Run the pipeline
  if (verbose) {
    llvm::outs() << "Running compilation pipeline...\n";
  }

  if (failed(pm.run(*module))) {
    llvm::errs() << "Compilation pipeline failed\n";
    return 1;
  }

  // Write output
  if (verbose) {
    llvm::outs() << "Writing output to: " << outputFilename << "\n";
  }

  if (failed(writeOutput(*module, outputFilename))) {
    llvm::errs() << "Failed to write output\n";
    return 1;
  }

  if (verbose) {
    llvm::outs() << "Compilation successful!\n";
  }

  return 0;
}
