//===----------------------------------------------------------------------===//
// Boas Dialect Implementation
//===----------------------------------------------------------------------===//

#include "mlirops/BoasDialect/BoasDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::boas;

//===----------------------------------------------------------------------===//
// Boas Dialect
//===----------------------------------------------------------------------===//

void BoasDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlirops/BoasDialect/BoasOps.cpp.inc"
  >();
  
  addTypes<TensorType, MatrixType>();
  addAttributes<NPUOptimizationAttr>();
}

BoasDialect::BoasDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<BoasDialect>()) {
  initialize();
}

//===----------------------------------------------------------------------===//
// Boas Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace boas {
namespace detail {

/// TensorType存储结构
struct TensorTypeStorage : public TypeStorage {
  TensorTypeStorage(ArrayRef<int64_t> shape, Type elementType, StringRef device)
      : shape(shape), elementType(elementType), device(device) {}

  using KeyTy = std::tuple<ArrayRef<int64_t>, Type, StringRef>;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == shape && std::get<1>(key) == elementType &&
           std::get<2>(key) == device;
  }

  static TensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    auto shapeArray = allocator.copyInto(std::get<0>(key));
    auto deviceStr = allocator.copyInto(std::get<2>(key));
    return new (allocator.allocate<TensorTypeStorage>())
        TensorTypeStorage(shapeArray, std::get<1>(key), deviceStr);
  }

  ArrayRef<int64_t> shape;
  Type elementType;
  StringRef device;
};

/// MatrixType存储结构
struct MatrixTypeStorage : public TypeStorage {
  MatrixTypeStorage(int64_t rows, int64_t cols, Type elementType, StringRef device)
      : rows(rows), cols(cols), elementType(elementType), device(device) {}

  using KeyTy = std::tuple<int64_t, int64_t, Type, StringRef>;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == rows && std::get<1>(key) == cols &&
           std::get<2>(key) == elementType && std::get<3>(key) == device;
  }

  static MatrixTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    auto deviceStr = allocator.copyInto(std::get<3>(key));
    return new (allocator.allocate<MatrixTypeStorage>())
        MatrixTypeStorage(std::get<0>(key), std::get<1>(key), 
                         std::get<2>(key), deviceStr);
  }

  int64_t rows, cols;
  Type elementType;
  StringRef device;
};

/// NPUOptimizationAttr存储结构
struct NPUOptimizationAttrStorage : public AttributeStorage {
  NPUOptimizationAttrStorage(IntegerAttr blockM, IntegerAttr blockN, IntegerAttr blockK,
                            BoolAttr useDiagonalTiling, StringRef strategy)
      : blockM(blockM), blockN(blockN), blockK(blockK),
        useDiagonalTiling(useDiagonalTiling), strategy(strategy) {}

  using KeyTy = std::tuple<IntegerAttr, IntegerAttr, IntegerAttr, BoolAttr, StringRef>;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == blockM && std::get<1>(key) == blockN &&
           std::get<2>(key) == blockK && std::get<3>(key) == useDiagonalTiling &&
           std::get<4>(key) == strategy;
  }

  static NPUOptimizationAttrStorage *construct(AttributeStorageAllocator &allocator,
                                              const KeyTy &key) {
    auto strategyStr = allocator.copyInto(std::get<4>(key));
    return new (allocator.allocate<NPUOptimizationAttrStorage>())
        NPUOptimizationAttrStorage(std::get<0>(key), std::get<1>(key), 
                                  std::get<2>(key), std::get<3>(key), strategyStr);
  }

  IntegerAttr blockM, blockN, blockK;
  BoolAttr useDiagonalTiling;
  StringRef strategy;
};

} // namespace detail
} // namespace boas
} // namespace mlir

//===----------------------------------------------------------------------===//
// TensorType Implementation
//===----------------------------------------------------------------------===//

TensorType TensorType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                          Type elementType, StringRef device) {
  return Base::get(context, shape, elementType, device);
}

ArrayRef<int64_t> TensorType::getShape() const {
  return getImpl()->shape;
}

Type TensorType::getElementType() const {
  return getImpl()->elementType;
}

StringRef TensorType::getDevice() const {
  return getImpl()->device;
}

bool TensorType::hasDynamicShape() const {
  return llvm::any_of(getShape(), [](int64_t dim) { return dim < 0; });
}

int64_t TensorType::getNumElements() const {
  int64_t numElements = 1;
  for (int64_t dim : getShape()) {
    if (dim < 0) return -1; // 动态形状
    numElements *= dim;
  }
  return numElements;
}

//===----------------------------------------------------------------------===//
// MatrixType Implementation
//===----------------------------------------------------------------------===//

MatrixType MatrixType::get(MLIRContext *context, int64_t rows, int64_t cols,
                          Type elementType, StringRef device) {
  return Base::get(context, rows, cols, elementType, device);
}

int64_t MatrixType::getRows() const {
  return getImpl()->rows;
}

int64_t MatrixType::getCols() const {
  return getImpl()->cols;
}

Type MatrixType::getElementType() const {
  return getImpl()->elementType;
}

StringRef MatrixType::getDevice() const {
  return getImpl()->device;
}

//===----------------------------------------------------------------------===//
// NPUOptimizationAttr Implementation
//===----------------------------------------------------------------------===//

NPUOptimizationAttr NPUOptimizationAttr::get(MLIRContext *context,
                                             IntegerAttr blockM, IntegerAttr blockN, 
                                             IntegerAttr blockK, BoolAttr useDiagonalTiling,
                                             StringAttr strategy) {
  return Base::get(context, blockM, blockN, blockK, useDiagonalTiling, 
                  strategy.getValue());
}

IntegerAttr NPUOptimizationAttr::getBlockM() const {
  return getImpl()->blockM;
}

IntegerAttr NPUOptimizationAttr::getBlockN() const {
  return getImpl()->blockN;
}

IntegerAttr NPUOptimizationAttr::getBlockK() const {
  return getImpl()->blockK;
}

BoolAttr NPUOptimizationAttr::getUseDiagonalTiling() const {
  return getImpl()->useDiagonalTiling;
}

StringAttr NPUOptimizationAttr::getStrategy() const {
  return StringAttr::get(getContext(), getImpl()->strategy);
}

//===----------------------------------------------------------------------===//
// Type and Attribute Parsing/Printing
//===----------------------------------------------------------------------===//

Type BoasDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  if (parser.parseKeyword(&typeKeyword))
    return Type();

  if (typeKeyword == "tensor") {
    // 解析: !boas.tensor<1024x512xf32@npu>
    if (parser.parseLess())
      return Type();

    SmallVector<int64_t> shape;
    if (parser.parseCommaSeparatedList([&]() {
      int64_t dim;
      if (parser.parseInteger(dim))
        return failure();
      shape.push_back(dim);
      return success();
    }))
      return Type();

    if (parser.parseKeyword("x"))
      return Type();

    Type elementType;
    if (parser.parseType(elementType))
      return Type();

    if (parser.parseKeyword("@"))
      return Type();

    StringRef device;
    if (parser.parseKeyword(&device))
      return Type();

    if (parser.parseGreater())
      return Type();

    return TensorType::get(getContext(), shape, elementType, device);
  }

  if (typeKeyword == "matrix") {
    // 解析: !boas.matrix<1024x512xf32@npu>
    if (parser.parseLess())
      return Type();

    int64_t rows, cols;
    if (parser.parseInteger(rows) || parser.parseKeyword("x") ||
        parser.parseInteger(cols) || parser.parseKeyword("x"))
      return Type();

    Type elementType;
    if (parser.parseType(elementType))
      return Type();

    if (parser.parseKeyword("@"))
      return Type();

    StringRef device;
    if (parser.parseKeyword(&device))
      return Type();

    if (parser.parseGreater())
      return Type();

    return MatrixType::get(getContext(), rows, cols, elementType, device);
  }

  parser.emitError(parser.getNameLoc(), "unknown boas type: " + typeKeyword);
  return Type();
}

void BoasDialect::printType(Type type, DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
    .Case<TensorType>([&](TensorType tensorType) {
      printer << "tensor<";
      ArrayRef<int64_t> shape = tensorType.getShape();
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) printer << "x";
        printer << shape[i];
      }
      printer << "x" << tensorType.getElementType() 
              << "@" << tensorType.getDevice() << ">";
    })
    .Case<MatrixType>([&](MatrixType matrixType) {
      printer << "matrix<" << matrixType.getRows() << "x" 
              << matrixType.getCols() << "x" << matrixType.getElementType()
              << "@" << matrixType.getDevice() << ">";
    });
}

Attribute BoasDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  StringRef attrKeyword;
  if (parser.parseKeyword(&attrKeyword))
    return Attribute();

  if (attrKeyword == "npu_opt") {
    // 解析: #boas.npu_opt<128,256,256,true,"diagonal">
    if (parser.parseLess())
      return Attribute();

    IntegerAttr blockM, blockN, blockK;
    BoolAttr useDiagonalTiling;
    StringAttr strategy;

    if (parser.parseAttribute(blockM) || parser.parseComma() ||
        parser.parseAttribute(blockN) || parser.parseComma() ||
        parser.parseAttribute(blockK) || parser.parseComma() ||
        parser.parseAttribute(useDiagonalTiling) || parser.parseComma() ||
        parser.parseAttribute(strategy))
      return Attribute();

    if (parser.parseGreater())
      return Attribute();

    return NPUOptimizationAttr::get(getContext(), blockM, blockN, blockK,
                                   useDiagonalTiling, strategy);
  }

  parser.emitError(parser.getNameLoc(), "unknown boas attribute: " + attrKeyword);
  return Attribute();
}

void BoasDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  TypeSwitch<Attribute>(attr)
    .Case<NPUOptimizationAttr>([&](NPUOptimizationAttr npuAttr) {
      printer << "npu_opt<" << npuAttr.getBlockM() << ","
              << npuAttr.getBlockN() << "," << npuAttr.getBlockK() << ","
              << npuAttr.getUseDiagonalTiling() << "," 
              << npuAttr.getStrategy() << ">";
    });
}

//===----------------------------------------------------------------------===//
// Generated Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlirops/BoasDialect/BoasOps.cpp.inc"
