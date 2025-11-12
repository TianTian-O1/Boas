//===- BoasTypes.cpp - Boas dialect types ----------------------*- C++ -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#include "Boas/Dialect/Boas/IR/BoasTypes.h"
#include "Boas/Dialect/Boas/IR/BoasDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::boas;

//===----------------------------------------------------------------------===//
// Custom parser/printer for TensorType
//===----------------------------------------------------------------------===//

static ParseResult parseShapeAndType(AsmParser &parser,
                                      SmallVectorImpl<int64_t> &shape,
                                      Type &elementType) {
  // Parse shape dimensions like: 2x3xf32
  // First, parse all dimensions
  do {
    int64_t dim;
    auto optionalInt = parser.parseOptionalInteger(dim);
    if (optionalInt.has_value()) {
      if (failed(optionalInt.value()))
        return failure();
      shape.push_back(dim);
    } else {
      // Check for dynamic dimension '?'
      if (succeeded(parser.parseOptionalQuestion())) {
        shape.push_back(mlir::ShapedType::kDynamic);
      } else {
        break;
      }
    }
  } while (succeeded(parser.parseOptionalKeyword("x")));

  // Parse element type
  if (parser.parseType(elementType))
    return failure();

  return success();
}

static void printShapeAndType(AsmPrinter &printer,
                               ArrayRef<int64_t> shape,
                               Type elementType) {
  // Print shape like: 2x3xf32
  for (int64_t i = 0, e = shape.size(); i < e; ++i) {
    if (i > 0)
      printer << 'x';
    if (mlir::ShapedType::isDynamic(shape[i]))
      printer << '?';
    else
      printer << shape[i];
  }
  printer << 'x' << elementType;
}

#define GET_TYPEDEF_CLASSES
#include "Boas/Dialect/Boas/IR/BoasTypes.cpp.inc"

void BoasDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Boas/Dialect/Boas/IR/BoasTypes.cpp.inc"
      >();
}
