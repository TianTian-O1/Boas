//===- BoasTypes.h - Boas dialect types -------------------------*- C++ -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#ifndef BOAS_DIALECT_BOAS_IR_BOASTYPES_H
#define BOAS_DIALECT_BOAS_IR_BOASTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Boas/Dialect/Boas/IR/BoasTypes.h.inc"

#endif // BOAS_DIALECT_BOAS_IR_BOASTYPES_H
