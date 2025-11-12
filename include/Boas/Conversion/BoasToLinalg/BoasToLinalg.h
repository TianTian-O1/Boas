//===- BoasToLinalg.h - Boas to Linalg conversion ---------------*- C++ -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//
//
// This file declares the pass for converting Boas dialect operations to
// Linalg dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef BOAS_CONVERSION_BOASTOLINALG_BOASTOLINALG_H
#define BOAS_CONVERSION_BOASTOLINALG_BOASTOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace boas {

/// Create a pass to convert Boas dialect operations to Linalg dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertBoasToLinalgPass();

} // namespace boas
} // namespace mlir

#endif // BOAS_CONVERSION_BOASTOLINALG_BOASTOLINALG_H
