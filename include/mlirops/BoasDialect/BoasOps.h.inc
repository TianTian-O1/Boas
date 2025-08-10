/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: BoasOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace boas {
class DeviceSelectOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class MatmulOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class MatmulOptimizedOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class NPUKernelOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class NPULaunchOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class PrintOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class ShapeOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class TensorCreateOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class TensorRandomOp;
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {
class ToDeviceOp;
} // namespace boas
} // namespace mlir
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::DeviceSelectOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DeviceSelectOpGenericAdaptorBase {
public:
  struct Properties {
    using deviceTy = ::mlir::StringAttr;
    deviceTy device;

    auto getDevice() {
      auto &propStorage = this->device;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setDevice(const ::mlir::StringAttr &propValue) {
      this->device = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.device == this->device &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  DeviceSelectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.device.select", odsAttrs.getContext());
  }

  DeviceSelectOpGenericAdaptorBase(DeviceSelectOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getDeviceAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().device);
    return attr;
  }

  ::llvm::StringRef getDevice();
};
} // namespace detail
template <typename RangeT>
class DeviceSelectOpGenericAdaptor : public detail::DeviceSelectOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DeviceSelectOpGenericAdaptorBase;
public:
  DeviceSelectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DeviceSelectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : DeviceSelectOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  DeviceSelectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : DeviceSelectOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  DeviceSelectOpGenericAdaptor(RangeT values, const DeviceSelectOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = DeviceSelectOp, typename = std::enable_if_t<std::is_same_v<LateInst, DeviceSelectOp>>>
  DeviceSelectOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DeviceSelectOpAdaptor : public DeviceSelectOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DeviceSelectOpGenericAdaptor::DeviceSelectOpGenericAdaptor;
  DeviceSelectOpAdaptor(DeviceSelectOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class DeviceSelectOp : public ::mlir::Op<DeviceSelectOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeviceSelectOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DeviceSelectOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("device")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDeviceAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDeviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.device.select");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getDeviceAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().device);
  }

  ::llvm::StringRef getDevice();
  void setDeviceAttr(::mlir::StringAttr attr) {
    getProperties().device = attr;
  }

  void setDevice(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef device);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::DeviceSelectOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::MatmulOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MatmulOpGenericAdaptorBase {
public:
  struct Properties {
    using npu_optTy = ::mlir::boas::NPUOptimizationAttr;
    npu_optTy npu_opt;

    auto getNpuOpt() {
      auto &propStorage = this->npu_opt;
      return ::llvm::dyn_cast_or_null<::mlir::boas::NPUOptimizationAttr>(propStorage);
    }
    void setNpuOpt(const ::mlir::boas::NPUOptimizationAttr &propValue) {
      this->npu_opt = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.npu_opt == this->npu_opt &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  MatmulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.matmul", odsAttrs.getContext());
  }

  MatmulOpGenericAdaptorBase(MatmulOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::boas::NPUOptimizationAttr getNpuOptAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::boas::NPUOptimizationAttr>(getProperties().npu_opt);
    return attr;
  }

  ::std::optional<::mlir::boas::NPUOptimizationAttr> getNpuOpt();
};
} // namespace detail
template <typename RangeT>
class MatmulOpGenericAdaptor : public detail::MatmulOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MatmulOpGenericAdaptorBase;
public:
  MatmulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MatmulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MatmulOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  MatmulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : MatmulOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  MatmulOpGenericAdaptor(RangeT values, const MatmulOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MatmulOp, typename = std::enable_if_t<std::is_same_v<LateInst, MatmulOp>>>
  MatmulOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MatmulOpAdaptor : public MatmulOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MatmulOpGenericAdaptor::MatmulOpGenericAdaptor;
  MatmulOpAdaptor(MatmulOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MatmulOp : public ::mlir::Op<MatmulOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::boas::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatmulOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MatmulOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("npu_opt")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNpuOptAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNpuOptAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.matmul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::boas::NPUOptimizationAttr getNpuOptAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::boas::NPUOptimizationAttr>(getProperties().npu_opt);
  }

  ::std::optional<::mlir::boas::NPUOptimizationAttr> getNpuOpt();
  void setNpuOptAttr(::mlir::boas::NPUOptimizationAttr attr) {
    getProperties().npu_opt = attr;
  }

  ::mlir::Attribute removeNpuOptAttr() {
      auto &attr = getProperties().npu_opt;
      attr = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::boas::NPUOptimizationAttr npu_opt);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::boas::NPUOptimizationAttr npu_opt);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::boas::NPUOptimizationAttr npu_opt);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// 获取输出矩阵的形状
  std::pair<int64_t, int64_t> getResultShape();

  /// 检查是否启用NPU优化
  bool hasNPUOptimization() { return getNpuOpt().has_value(); }

  /// 获取优化策略
  StringRef getOptimizationStrategy();
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::MatmulOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::MatmulOptimizedOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MatmulOptimizedOpGenericAdaptorBase {
public:
  struct Properties {
    using configTy = ::mlir::boas::NPUOptimizationAttr;
    configTy config;

    auto getConfig() {
      auto &propStorage = this->config;
      return ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(propStorage);
    }
    void setConfig(const ::mlir::boas::NPUOptimizationAttr &propValue) {
      this->config = propValue;
    }
    using strategyTy = ::mlir::StringAttr;
    strategyTy strategy;

    auto getStrategy() {
      auto &propStorage = this->strategy;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setStrategy(const ::mlir::StringAttr &propValue) {
      this->strategy = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.config == this->config &&
        rhs.strategy == this->strategy &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  MatmulOptimizedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.matmul.optimized", odsAttrs.getContext());
  }

  MatmulOptimizedOpGenericAdaptorBase(MatmulOptimizedOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getStrategyAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().strategy);
    return attr;
  }

  ::llvm::StringRef getStrategy();
  ::mlir::boas::NPUOptimizationAttr getConfigAttr() {
    auto attr = ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(getProperties().config);
    return attr;
  }

  ::mlir::boas::NPUOptimizationAttr getConfig();
};
} // namespace detail
template <typename RangeT>
class MatmulOptimizedOpGenericAdaptor : public detail::MatmulOptimizedOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MatmulOptimizedOpGenericAdaptorBase;
public:
  MatmulOptimizedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MatmulOptimizedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MatmulOptimizedOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  MatmulOptimizedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : MatmulOptimizedOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  MatmulOptimizedOpGenericAdaptor(RangeT values, const MatmulOptimizedOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MatmulOptimizedOp, typename = std::enable_if_t<std::is_same_v<LateInst, MatmulOptimizedOp>>>
  MatmulOptimizedOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MatmulOptimizedOpAdaptor : public MatmulOptimizedOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MatmulOptimizedOpGenericAdaptor::MatmulOptimizedOpGenericAdaptor;
  MatmulOptimizedOpAdaptor(MatmulOptimizedOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MatmulOptimizedOp : public ::mlir::Op<MatmulOptimizedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::boas::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatmulOptimizedOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MatmulOptimizedOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("config"), ::llvm::StringRef("strategy")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getConfigAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStrategyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStrategyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.matmul.optimized");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getStrategyAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().strategy);
  }

  ::llvm::StringRef getStrategy();
  ::mlir::boas::NPUOptimizationAttr getConfigAttr() {
    return ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(getProperties().config);
  }

  ::mlir::boas::NPUOptimizationAttr getConfig();
  void setStrategyAttr(::mlir::StringAttr attr) {
    getProperties().strategy = attr;
  }

  void setStrategy(::llvm::StringRef attrValue);
  void setConfigAttr(::mlir::boas::NPUOptimizationAttr attr) {
    getProperties().config = attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::StringAttr strategy, ::mlir::boas::NPUOptimizationAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::StringAttr strategy, ::mlir::boas::NPUOptimizationAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, ::llvm::StringRef strategy, ::mlir::boas::NPUOptimizationAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::llvm::StringRef strategy, ::mlir::boas::NPUOptimizationAttr config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::MatmulOptimizedOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::NPUKernelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class NPUKernelOpGenericAdaptorBase {
public:
  struct Properties {
    using block_sizeTy = ::mlir::IntegerAttr;
    block_sizeTy block_size;

    auto getBlockSize() {
      auto &propStorage = this->block_size;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setBlockSize(const ::mlir::IntegerAttr &propValue) {
      this->block_size = propValue;
    }
    using configTy = ::mlir::boas::NPUOptimizationAttr;
    configTy config;

    auto getConfig() {
      auto &propStorage = this->config;
      return ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(propStorage);
    }
    void setConfig(const ::mlir::boas::NPUOptimizationAttr &propValue) {
      this->config = propValue;
    }
    using grid_sizeTy = ::mlir::IntegerAttr;
    grid_sizeTy grid_size;

    auto getGridSize() {
      auto &propStorage = this->grid_size;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setGridSize(const ::mlir::IntegerAttr &propValue) {
      this->grid_size = propValue;
    }
    using kernel_nameTy = ::mlir::StringAttr;
    kernel_nameTy kernel_name;

    auto getKernelName() {
      auto &propStorage = this->kernel_name;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setKernelName(const ::mlir::StringAttr &propValue) {
      this->kernel_name = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.block_size == this->block_size &&
        rhs.config == this->config &&
        rhs.grid_size == this->grid_size &&
        rhs.kernel_name == this->kernel_name &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  NPUKernelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.npu.kernel", odsAttrs.getContext());
  }

  NPUKernelOpGenericAdaptorBase(NPUKernelOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getKernelNameAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().kernel_name);
    return attr;
  }

  ::llvm::StringRef getKernelName();
  ::mlir::boas::NPUOptimizationAttr getConfigAttr() {
    auto attr = ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(getProperties().config);
    return attr;
  }

  ::mlir::boas::NPUOptimizationAttr getConfig();
  ::mlir::IntegerAttr getGridSizeAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().grid_size);
    return attr;
  }

  ::std::optional<uint64_t> getGridSize();
  ::mlir::IntegerAttr getBlockSizeAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().block_size);
    return attr;
  }

  ::std::optional<uint64_t> getBlockSize();
  ::mlir::Region &getBody() {
    return *odsRegions[0];
  }

  ::mlir::RegionRange getRegions() {
    return odsRegions;
  }

};
} // namespace detail
template <typename RangeT>
class NPUKernelOpGenericAdaptor : public detail::NPUKernelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::NPUKernelOpGenericAdaptorBase;
public:
  NPUKernelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  NPUKernelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : NPUKernelOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  NPUKernelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : NPUKernelOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  NPUKernelOpGenericAdaptor(RangeT values, const NPUKernelOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = NPUKernelOp, typename = std::enable_if_t<std::is_same_v<LateInst, NPUKernelOp>>>
  NPUKernelOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return getODSOperands(0);
  }

private:
  RangeT odsOperands;
};
class NPUKernelOpAdaptor : public NPUKernelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using NPUKernelOpGenericAdaptor::NPUKernelOpGenericAdaptor;
  NPUKernelOpAdaptor(NPUKernelOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class NPUKernelOp : public ::mlir::Op<NPUKernelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = NPUKernelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = NPUKernelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("block_size"), ::llvm::StringRef("config"), ::llvm::StringRef("grid_size"), ::llvm::StringRef("kernel_name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBlockSizeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBlockSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getConfigAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getConfigAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getGridSizeAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getGridSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getKernelNameAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getKernelNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.npu.kernel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getOperands() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getResults() {
    return getODSResults(0);
  }

  ::mlir::Region &getBody() {
    return (*this)->getRegion(0);
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getKernelNameAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().kernel_name);
  }

  ::llvm::StringRef getKernelName();
  ::mlir::boas::NPUOptimizationAttr getConfigAttr() {
    return ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(getProperties().config);
  }

  ::mlir::boas::NPUOptimizationAttr getConfig();
  ::mlir::IntegerAttr getGridSizeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().grid_size);
  }

  ::std::optional<uint64_t> getGridSize();
  ::mlir::IntegerAttr getBlockSizeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().block_size);
  }

  ::std::optional<uint64_t> getBlockSize();
  void setKernelNameAttr(::mlir::StringAttr attr) {
    getProperties().kernel_name = attr;
  }

  void setKernelName(::llvm::StringRef attrValue);
  void setConfigAttr(::mlir::boas::NPUOptimizationAttr attr) {
    getProperties().config = attr;
  }

  void setGridSizeAttr(::mlir::IntegerAttr attr) {
    getProperties().grid_size = attr;
  }

  void setGridSize(::std::optional<uint64_t> attrValue);
  void setBlockSizeAttr(::mlir::IntegerAttr attr) {
    getProperties().block_size = attr;
  }

  void setBlockSize(::std::optional<uint64_t> attrValue);
  ::mlir::Attribute removeGridSizeAttr() {
      auto &attr = getProperties().grid_size;
      attr = {};
      return attr;
  }

  ::mlir::Attribute removeBlockSizeAttr() {
      auto &attr = getProperties().block_size;
      attr = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange operands, ::mlir::StringAttr kernel_name, ::mlir::boas::NPUOptimizationAttr config, /*optional*/::mlir::IntegerAttr grid_size, /*optional*/::mlir::IntegerAttr block_size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange operands, ::llvm::StringRef kernel_name, ::mlir::boas::NPUOptimizationAttr config, /*optional*/::mlir::IntegerAttr grid_size, /*optional*/::mlir::IntegerAttr block_size);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::NPUKernelOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::NPULaunchOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class NPULaunchOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  NPULaunchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.npu.launch", odsAttrs.getContext());
  }

  NPULaunchOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::Region &getBody() {
    return *odsRegions[0];
  }

  ::mlir::RegionRange getRegions() {
    return odsRegions;
  }

};
} // namespace detail
template <typename RangeT>
class NPULaunchOpGenericAdaptor : public detail::NPULaunchOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::NPULaunchOpGenericAdaptorBase;
public:
  NPULaunchOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  NPULaunchOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : NPULaunchOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  NPULaunchOpGenericAdaptor(RangeT values, const NPULaunchOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = NPULaunchOp, typename = std::enable_if_t<std::is_same_v<LateInst, NPULaunchOp>>>
  NPULaunchOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getGridX() {
    return (*getODSOperands(0).begin());
  }

  ValueT getGridY() {
    return (*getODSOperands(1).begin());
  }

  ValueT getGridZ() {
    return (*getODSOperands(2).begin());
  }

  ValueT getBlockX() {
    return (*getODSOperands(3).begin());
  }

  ValueT getBlockY() {
    return (*getODSOperands(4).begin());
  }

  ValueT getBlockZ() {
    return (*getODSOperands(5).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class NPULaunchOpAdaptor : public NPULaunchOpGenericAdaptor<::mlir::ValueRange> {
public:
  using NPULaunchOpGenericAdaptor::NPULaunchOpGenericAdaptor;
  NPULaunchOpAdaptor(NPULaunchOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class NPULaunchOp : public ::mlir::Op<NPULaunchOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = NPULaunchOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = NPULaunchOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.npu.launch");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IndexType> getGridX() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IndexType> getGridY() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IndexType> getGridZ() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IndexType> getBlockX() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(3).begin());
  }

  ::mlir::TypedValue<::mlir::IndexType> getBlockY() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(4).begin());
  }

  ::mlir::TypedValue<::mlir::IndexType> getBlockZ() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(5).begin());
  }

  ::mlir::OpOperand &getGridXMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getGridYMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getGridZMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBlockXMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBlockYMutable() {
    auto range = getODSOperandIndexAndLength(4);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBlockZMutable() {
    auto range = getODSOperandIndexAndLength(5);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Region &getBody() {
    return (*this)->getRegion(0);
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value gridX, ::mlir::Value gridY, ::mlir::Value gridZ, ::mlir::Value blockX, ::mlir::Value blockY, ::mlir::Value blockZ);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value gridX, ::mlir::Value gridY, ::mlir::Value gridZ, ::mlir::Value blockX, ::mlir::Value blockY, ::mlir::Value blockZ);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::NPULaunchOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::PrintOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class PrintOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  PrintOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.print", odsAttrs.getContext());
  }

  PrintOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class PrintOpGenericAdaptor : public detail::PrintOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PrintOpGenericAdaptorBase;
public:
  PrintOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  PrintOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : PrintOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  PrintOpGenericAdaptor(RangeT values, const PrintOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = PrintOp, typename = std::enable_if_t<std::is_same_v<LateInst, PrintOp>>>
  PrintOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PrintOpAdaptor : public PrintOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PrintOpGenericAdaptor::PrintOpGenericAdaptor;
  PrintOpAdaptor(PrintOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class PrintOp : public ::mlir::Op<PrintOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PrintOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PrintOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.print");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::PrintOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::ShapeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ShapeOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ShapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.shape", odsAttrs.getContext());
  }

  ShapeOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class ShapeOpGenericAdaptor : public detail::ShapeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ShapeOpGenericAdaptorBase;
public:
  ShapeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ShapeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ShapeOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ShapeOpGenericAdaptor(RangeT values, const ShapeOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ShapeOp, typename = std::enable_if_t<std::is_same_v<LateInst, ShapeOp>>>
  ShapeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ShapeOpAdaptor : public ShapeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ShapeOpGenericAdaptor::ShapeOpGenericAdaptor;
  ShapeOpAdaptor(ShapeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ShapeOp : public ::mlir::Op<ShapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::NResults<2>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShapeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ShapeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.shape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IndexType> getRows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(0).begin());
  }

  ::mlir::TypedValue<::mlir::IndexType> getCols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(1).begin());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type rows, ::mlir::Type cols, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::ShapeOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::TensorCreateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TensorCreateOpGenericAdaptorBase {
public:
  struct Properties {
    using deviceTy = ::mlir::StringAttr;
    deviceTy device;

    auto getDevice() {
      auto &propStorage = this->device;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setDevice(const ::mlir::StringAttr &propValue) {
      this->device = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.device == this->device &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TensorCreateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.tensor.create", odsAttrs.getContext());
  }

  TensorCreateOpGenericAdaptorBase(TensorCreateOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getDeviceAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().device);
    return attr;
  }

  ::llvm::StringRef getDevice();
};
} // namespace detail
template <typename RangeT>
class TensorCreateOpGenericAdaptor : public detail::TensorCreateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TensorCreateOpGenericAdaptorBase;
public:
  TensorCreateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TensorCreateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TensorCreateOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TensorCreateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TensorCreateOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TensorCreateOpGenericAdaptor(RangeT values, const TensorCreateOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TensorCreateOp, typename = std::enable_if_t<std::is_same_v<LateInst, TensorCreateOp>>>
  TensorCreateOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRows() {
    return (*getODSOperands(0).begin());
  }

  ValueT getCols() {
    return (*getODSOperands(1).begin());
  }

  ValueT getValues() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TensorCreateOpAdaptor : public TensorCreateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TensorCreateOpGenericAdaptor::TensorCreateOpGenericAdaptor;
  TensorCreateOpAdaptor(TensorCreateOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TensorCreateOp : public ::mlir::Op<TensorCreateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::boas::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TensorCreateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TensorCreateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("device")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDeviceAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDeviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.tensor.create");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IndexType> getRows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IndexType> getCols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getValues() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getRowsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getColsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getValuesMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getDeviceAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().device);
  }

  ::llvm::StringRef getDevice();
  void setDeviceAttr(::mlir::StringAttr attr) {
    getProperties().device = attr;
  }

  void setDevice(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::mlir::StringAttr device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::mlir::StringAttr device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::mlir::StringAttr device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::llvm::StringRef device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::llvm::StringRef device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::llvm::StringRef device);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::TensorCreateOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::TensorRandomOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TensorRandomOpGenericAdaptorBase {
public:
  struct Properties {
    using deviceTy = ::mlir::StringAttr;
    deviceTy device;

    auto getDevice() {
      auto &propStorage = this->device;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setDevice(const ::mlir::StringAttr &propValue) {
      this->device = propValue;
    }
    using seedTy = ::mlir::FloatAttr;
    seedTy seed;

    auto getSeed() {
      auto &propStorage = this->seed;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setSeed(const ::mlir::FloatAttr &propValue) {
      this->seed = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.device == this->device &&
        rhs.seed == this->seed &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TensorRandomOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.tensor.random", odsAttrs.getContext());
  }

  TensorRandomOpGenericAdaptorBase(TensorRandomOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getDeviceAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().device);
    return attr;
  }

  ::llvm::StringRef getDevice();
  ::mlir::FloatAttr getSeedAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().seed);
    return attr;
  }

  ::std::optional< ::llvm::APFloat > getSeed();
};
} // namespace detail
template <typename RangeT>
class TensorRandomOpGenericAdaptor : public detail::TensorRandomOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TensorRandomOpGenericAdaptorBase;
public:
  TensorRandomOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TensorRandomOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TensorRandomOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TensorRandomOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TensorRandomOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TensorRandomOpGenericAdaptor(RangeT values, const TensorRandomOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TensorRandomOp, typename = std::enable_if_t<std::is_same_v<LateInst, TensorRandomOp>>>
  TensorRandomOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRows() {
    return (*getODSOperands(0).begin());
  }

  ValueT getCols() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TensorRandomOpAdaptor : public TensorRandomOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TensorRandomOpGenericAdaptor::TensorRandomOpGenericAdaptor;
  TensorRandomOpAdaptor(TensorRandomOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TensorRandomOp : public ::mlir::Op<TensorRandomOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::boas::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TensorRandomOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TensorRandomOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("device"), ::llvm::StringRef("seed")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDeviceAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDeviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSeedAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSeedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.tensor.random");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IndexType> getRows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IndexType> getCols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getRowsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getColsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getDeviceAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().device);
  }

  ::llvm::StringRef getDevice();
  ::mlir::FloatAttr getSeedAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().seed);
  }

  ::std::optional< ::llvm::APFloat > getSeed();
  void setDeviceAttr(::mlir::StringAttr attr) {
    getProperties().device = attr;
  }

  void setDevice(::llvm::StringRef attrValue);
  void setSeedAttr(::mlir::FloatAttr attr) {
    getProperties().seed = attr;
  }

  void setSeed(::std::optional<::llvm::APFloat> attrValue);
  ::mlir::Attribute removeSeedAttr() {
      auto &attr = getProperties().seed;
      attr = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value rows, ::mlir::Value cols, ::mlir::StringAttr device, /*optional*/::mlir::FloatAttr seed);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value rows, ::mlir::Value cols, ::mlir::StringAttr device, /*optional*/::mlir::FloatAttr seed);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value rows, ::mlir::Value cols, ::llvm::StringRef device, /*optional*/::mlir::FloatAttr seed);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value rows, ::mlir::Value cols, ::llvm::StringRef device, /*optional*/::mlir::FloatAttr seed);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::TensorRandomOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::ToDeviceOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ToDeviceOpGenericAdaptorBase {
public:
  struct Properties {
    using target_deviceTy = ::mlir::StringAttr;
    target_deviceTy target_device;

    auto getTargetDevice() {
      auto &propStorage = this->target_device;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setTargetDevice(const ::mlir::StringAttr &propValue) {
      this->target_device = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.target_device == this->target_device &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ToDeviceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("boas.to_device", odsAttrs.getContext());
  }

  ToDeviceOpGenericAdaptorBase(ToDeviceOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getTargetDeviceAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().target_device);
    return attr;
  }

  ::llvm::StringRef getTargetDevice();
};
} // namespace detail
template <typename RangeT>
class ToDeviceOpGenericAdaptor : public detail::ToDeviceOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ToDeviceOpGenericAdaptorBase;
public:
  ToDeviceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ToDeviceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ToDeviceOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ToDeviceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ToDeviceOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ToDeviceOpGenericAdaptor(RangeT values, const ToDeviceOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ToDeviceOp, typename = std::enable_if_t<std::is_same_v<LateInst, ToDeviceOp>>>
  ToDeviceOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ToDeviceOpAdaptor : public ToDeviceOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ToDeviceOpGenericAdaptor::ToDeviceOpGenericAdaptor;
  ToDeviceOpAdaptor(ToDeviceOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ToDeviceOp : public ::mlir::Op<ToDeviceOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::boas::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ToDeviceOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ToDeviceOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("target_device")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getTargetDeviceAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTargetDeviceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("boas.to_device");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::boas::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::boas::TensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getTargetDeviceAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().target_device);
  }

  ::llvm::StringRef getTargetDevice();
  void setTargetDeviceAttr(::mlir::StringAttr attr) {
    getProperties().target_device = attr;
  }

  void setTargetDevice(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::StringAttr target_device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::StringAttr target_device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::llvm::StringRef target_device);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::StringRef target_device);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace boas
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::boas::ToDeviceOp)


#endif  // GET_OP_CLASSES

