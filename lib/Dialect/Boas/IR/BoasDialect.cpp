//===- BoasDialect.cpp - Boas dialect ---------------------------*- C++ -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#include "Boas/Dialect/Boas/IR/BoasDialect.h"
#include "Boas/Dialect/Boas/IR/BoasOps.h"
#include "Boas/Dialect/Boas/IR/BoasTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::boas;

//===----------------------------------------------------------------------===//
// Boas dialect
//===----------------------------------------------------------------------===//

void BoasDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Boas/Dialect/Boas/IR/BoasOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Boas/Dialect/Boas/IR/BoasTypes.cpp.inc"
      >();
}

#include "Boas/Dialect/Boas/IR/BoasDialect.cpp.inc"
