//===- BoasDialect.td - Boas dialect definition -----------*- tablegen -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the Boas dialect and its base classes.
//
//===----------------------------------------------------------------------===//

#ifndef BOAS_DIALECT_BOAS_IR_BOASDIALECT_TD
#define BOAS_DIALECT_BOAS_IR_BOASDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Boas Dialect Definition
//===----------------------------------------------------------------------===//

def Boas_Dialect : Dialect {
  let name = "boas";
  let summary = "Boas programming language dialect";
  let description = [{
    The Boas dialect provides high-level operations for the Boas programming
    language, which combines Python-like syntax with NPU acceleration capabilities.

    This dialect serves as the entry point for Boas programs and provides
    operations for:
    - Tensor creation and manipulation
    - Matrix operations
    - NPU-specific constructs
    - Control flow
    - I/O operations

    The Boas dialect is designed to lower to standard MLIR dialects (Linalg, Tensor)
    and eventually to BiShengIR dialects (HFusion, HIVM) for NPU execution.
  }];

  let cppNamespace = "::mlir::boas";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

// Base class for Boas dialect operations
class Boas_Op<string mnemonic, list<Trait> traits = []> :
    Op<Boas_Dialect, mnemonic, traits>;

#endif // BOAS_DIALECT_BOAS_IR_BOASDIALECT_TD
