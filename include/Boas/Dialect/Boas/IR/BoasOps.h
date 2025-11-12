//===- BoasOps.h - Boas dialect ops -----------------------------*- C++ -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#ifndef BOAS_DIALECT_BOAS_IR_BOASOPS_H
#define BOAS_DIALECT_BOAS_IR_BOASOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Boas/Dialect/Boas/IR/BoasDialect.h"
#include "Boas/Dialect/Boas/IR/BoasTypes.h"

#define GET_OP_CLASSES
#include "Boas/Dialect/Boas/IR/BoasOps.h.inc"

#endif // BOAS_DIALECT_BOAS_IR_BOASOPS_H
