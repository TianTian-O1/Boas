//===- BoasTypes.td - Boas type definitions ---------------*- tablegen -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#ifndef BOAS_DIALECT_BOAS_IR_BOASTYPES_TD
#define BOAS_DIALECT_BOAS_IR_BOASTYPES_TD

include "Boas/Dialect/Boas/IR/BoasDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Boas Type Definitions
//===----------------------------------------------------------------------===//

class Boas_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Boas_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

def Boas_TensorType : Boas_Type<"Tensor", "tensor"> {
  let summary = "Boas tensor type";
  let description = [{
    Represents a multi-dimensional tensor in Boas language.
    Similar to MLIR's RankedTensorType but with Boas-specific semantics.

    Syntax:
      !boas.tensor<2x3xf32>
      !boas.tensor<4x5x6xi32>
      !boas.tensor<?x10xf32>  // dynamic dimension
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let assemblyFormat = "`<` custom<ShapeAndType>($shape, $elementType) `>`";

  let extraClassDeclaration = [{
    /// Returns the rank of the tensor
    int64_t getRank() const { return getShape().size(); }

    /// Check if any dimension is dynamic
    bool hasDynamicDim() const {
      return llvm::any_of(getShape(), [](int64_t dim) {
        return mlir::ShapedType::isDynamic(dim);
      });
    }

    /// Get number of elements (product of all dimensions)
    int64_t getNumElements() const {
      if (hasDynamicDim())
        return mlir::ShapedType::kDynamic;
      int64_t result = 1;
      for (auto dim : getShape())
        result *= dim;
      return result;
    }
  }];
}

//===----------------------------------------------------------------------===//
// DeviceType
//===----------------------------------------------------------------------===//

def Boas_DeviceType : Boas_Type<"Device", "device"> {
  let summary = "NPU device type";
  let description = [{
    Represents an NPU device in Boas language.
    Used for device management and placement.

    Syntax:
      !boas.device
  }];

  let parameters = (ins
    "int32_t":$deviceId
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins "int32_t":$deviceId), [{
      return $_get($_ctxt, deviceId);
    }]>
  ];

  let assemblyFormat = "`<` $deviceId `>`";
}

//===----------------------------------------------------------------------===//
// Type Predicates
//===----------------------------------------------------------------------===//

def Boas_Tensor : Type<
  CPred<"$_self.isa<::mlir::boas::TensorType>()">,
  "Boas tensor type">;

def Boas_Device : Type<
  CPred<"$_self.isa<::mlir::boas::DeviceType>()">,
  "Boas device type">;

#endif // BOAS_DIALECT_BOAS_IR_BOASTYPES_TD
