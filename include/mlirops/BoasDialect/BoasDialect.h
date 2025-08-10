#ifndef BOAS_DIALECT_H
#define BOAS_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// Boas Dialect
//===----------------------------------------------------------------------===//

/// Boas语言专用MLIR方言
/// 提供高级语义操作，支持矩阵运算、张量操作和NPU优化
class BoasDialect : public Dialect {
public:
  explicit BoasDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "boas"; }

  /// 注册Boas方言中的类型
  void initialize();

  /// 解析属性
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  /// 打印属性
  void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;

  /// 解析类型
  Type parseType(DialectAsmParser &parser) const override;

  /// 打印类型
  void printType(Type type, DialectAsmPrinter &printer) const override;
};

//===----------------------------------------------------------------------===//
// Boas Types
//===----------------------------------------------------------------------===//

/// Boas张量类型
/// 封装了形状、数据类型和设备信息
class TensorType : public Type::TypeBase<TensorType, Type, detail::TensorTypeStorage> {
public:
  using Base::Base;

  /// 创建张量类型
  static TensorType get(MLIRContext *context, ArrayRef<int64_t> shape, 
                       Type elementType, StringRef device = "cpu");

  /// 获取张量形状
  ArrayRef<int64_t> getShape() const;

  /// 获取元素类型
  Type getElementType() const;

  /// 获取设备信息
  StringRef getDevice() const;

  /// 检查是否为动态形状
  bool hasDynamicShape() const;

  /// 获取总元素数量
  int64_t getNumElements() const;
};

/// Boas矩阵类型（特化的张量类型）
class MatrixType : public Type::TypeBase<MatrixType, Type, detail::MatrixTypeStorage> {
public:
  using Base::Base;

  /// 创建矩阵类型
  static MatrixType get(MLIRContext *context, int64_t rows, int64_t cols, 
                       Type elementType, StringRef device = "cpu");

  /// 获取行数
  int64_t getRows() const;

  /// 获取列数
  int64_t getCols() const;

  /// 获取元素类型
  Type getElementType() const;

  /// 获取设备信息
  StringRef getDevice() const;
};

//===----------------------------------------------------------------------===//
// Boas Attributes
//===----------------------------------------------------------------------===//

/// NPU优化属性
/// 包含块配置、分核策略等NPU特定优化信息
class NPUOptimizationAttr : public Attribute::AttrBase<NPUOptimizationAttr, Attribute,
                                                       detail::NPUOptimizationAttrStorage> {
public:
  using Base::Base;

  /// 创建NPU优化属性
  static NPUOptimizationAttr get(MLIRContext *context,
                                IntegerAttr blockM, IntegerAttr blockN, IntegerAttr blockK,
                                BoolAttr useDiagonalTiling, StringAttr strategy);

  /// 获取块配置
  IntegerAttr getBlockM() const;
  IntegerAttr getBlockN() const;
  IntegerAttr getBlockK() const;

  /// 获取分核策略
  BoolAttr getUseDiagonalTiling() const;
  StringAttr getStrategy() const;
};

//===----------------------------------------------------------------------===//
// Boas Operations - 前向声明
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlirops/BoasDialect/BoasOps.h.inc"

} // namespace boas
} // namespace mlir

#endif // BOAS_DIALECT_H
