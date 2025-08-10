/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: BoasOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::boas::DeviceSelectOp,
::mlir::boas::MatmulOp,
::mlir::boas::MatmulOptimizedOp,
::mlir::boas::NPUKernelOp,
::mlir::boas::NPULaunchOp,
::mlir::boas::PrintOp,
::mlir::boas::ShapeOp,
::mlir::boas::TensorCreateOp,
::mlir::boas::TensorRandomOp,
::mlir::boas::ToDeviceOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace boas {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_BoasOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::boas::TensorType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Boas\345\274\240\351\207\217\347\261\273\345\236\213, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_BoasOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_BoasOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::IndexType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_BoasOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_BoasOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || (((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values or tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_BoasOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_BoasOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_BoasOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_BoasOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::boas::NPUOptimizationAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: NPU\344\274\230\345\214\226\351\205\215\347\275\256\345\261\236\346\200\247";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_BoasOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_BoasOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_BoasOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_BoasOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_BoasOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_BoasOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::FloatAttr>(attr))) && ((::llvm::cast<::mlir::FloatAttr>(attr).getType().isF64()))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit float attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_BoasOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_BoasOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_region_constraint_BoasOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace boas
} // namespace mlir
namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::DeviceSelectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DeviceSelectOpGenericAdaptorBase::DeviceSelectOpGenericAdaptorBase(DeviceSelectOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef DeviceSelectOpGenericAdaptorBase::getDevice() {
  auto attr = getDeviceAttr();
  return attr.getValue();
}

} // namespace detail
DeviceSelectOpAdaptor::DeviceSelectOpAdaptor(DeviceSelectOp op) : DeviceSelectOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult DeviceSelectOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_device = getProperties().device; (void)tblgen_device;
  if (!tblgen_device) return emitError(loc, "'boas.device.select' op ""requires attribute 'device'");

  if (tblgen_device && !((::llvm::isa<::mlir::StringAttr>(tblgen_device))))
    return emitError(loc, "'boas.device.select' op ""attribute 'device' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult DeviceSelectOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.device;
       auto attr = dict.get("device");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `device` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute DeviceSelectOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.device;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("device",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code DeviceSelectOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.device.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> DeviceSelectOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "device")
      return prop.device;
  return std::nullopt;
}

void DeviceSelectOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "device") {
       prop.device = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.device)>>(value);
       return;
    }
}

void DeviceSelectOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.device) attrs.append("device", prop.device);
}

::llvm::LogicalResult DeviceSelectOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getDeviceAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(attr, "device", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult DeviceSelectOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.device)))
    return ::mlir::failure();
  return ::mlir::success();
}

void DeviceSelectOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.device);
}

::llvm::StringRef DeviceSelectOp::getDevice() {
  auto attr = getDeviceAttr();
  return attr.getValue();
}

void DeviceSelectOp::setDevice(::llvm::StringRef attrValue) {
  getProperties().device = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void DeviceSelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr device) {
  odsState.getOrAddProperties<Properties>().device = device;
}

void DeviceSelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr device) {
  odsState.getOrAddProperties<Properties>().device = device;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeviceSelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef device) {
  odsState.getOrAddProperties<Properties>().device = odsBuilder.getStringAttr(device);
}

void DeviceSelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef device) {
  odsState.getOrAddProperties<Properties>().device = odsBuilder.getStringAttr(device);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeviceSelectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<DeviceSelectOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult DeviceSelectOp::verifyInvariantsImpl() {
  auto tblgen_device = getProperties().device; (void)tblgen_device;
  if (!tblgen_device) return emitOpError("requires attribute 'device'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(*this, tblgen_device, "device")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult DeviceSelectOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DeviceSelectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr deviceAttr;

  if (parser.parseCustomAttributeWithFallback(deviceAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (deviceAttr) result.getOrAddProperties<DeviceSelectOp::Properties>().device = deviceAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void DeviceSelectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getDeviceAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("device");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::DeviceSelectOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::MatmulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MatmulOpGenericAdaptorBase::MatmulOpGenericAdaptorBase(MatmulOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::std::optional<::mlir::boas::NPUOptimizationAttr> MatmulOpGenericAdaptorBase::getNpuOpt() {
  auto attr = getNpuOptAttr();
  return attr ? ::std::optional<::mlir::boas::NPUOptimizationAttr>(::llvm::cast<::mlir::boas::NPUOptimizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
MatmulOpAdaptor::MatmulOpAdaptor(MatmulOp op) : MatmulOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult MatmulOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_npu_opt = getProperties().npu_opt; (void)tblgen_npu_opt;

  if (tblgen_npu_opt && !((::llvm::isa<::mlir::boas::NPUOptimizationAttr>(tblgen_npu_opt))))
    return emitError(loc, "'boas.matmul' op ""attribute 'npu_opt' failed to satisfy constraint: NPU\344\274\230\345\214\226\351\205\215\347\275\256\345\261\236\346\200\247");
  return ::mlir::success();
}

::llvm::LogicalResult MatmulOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.npu_opt;
       auto attr = dict.get("npu_opt");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `npu_opt` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute MatmulOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.npu_opt;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("npu_opt",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code MatmulOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.npu_opt.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> MatmulOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "npu_opt")
      return prop.npu_opt;
  return std::nullopt;
}

void MatmulOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "npu_opt") {
       prop.npu_opt = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.npu_opt)>>(value);
       return;
    }
}

void MatmulOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.npu_opt) attrs.append("npu_opt", prop.npu_opt);
}

::llvm::LogicalResult MatmulOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getNpuOptAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps2(attr, "npu_opt", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult MatmulOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.npu_opt)))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatmulOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.npu_opt);
}

::std::optional<::mlir::boas::NPUOptimizationAttr> MatmulOp::getNpuOpt() {
  auto attr = getNpuOptAttr();
  return attr ? ::std::optional<::mlir::boas::NPUOptimizationAttr>(::llvm::cast<::mlir::boas::NPUOptimizationAttr>(attr)) : (::std::nullopt);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::boas::NPUOptimizationAttr npu_opt) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (npu_opt) {
    odsState.getOrAddProperties<Properties>().npu_opt = npu_opt;
  }
  odsState.addTypes(result);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::boas::NPUOptimizationAttr npu_opt) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (npu_opt) {
    odsState.getOrAddProperties<Properties>().npu_opt = npu_opt;
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MatmulOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::boas::NPUOptimizationAttr npu_opt) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (npu_opt) {
    odsState.getOrAddProperties<Properties>().npu_opt = npu_opt;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatmulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<MatmulOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<MatmulOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(MatmulOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::llvm::LogicalResult MatmulOp::verifyInvariantsImpl() {
  auto tblgen_npu_opt = getProperties().npu_opt; (void)tblgen_npu_opt;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps2(*this, tblgen_npu_opt, "npu_opt")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that lhs and rhs have compatible shapes");
  if (!((std::equal_to<>()((*this->getODSOperands(0).begin()).getType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result shape matches matmul result");
  return ::mlir::success();
}

::llvm::LogicalResult MatmulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult MatmulOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  if (operands.size() <= 0)
    return ::mlir::failure();
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult MatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(&lhsRawOperand, 1);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(&rhsRawOperand, 1);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawType{};
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(&lhsRawType, 1);
  ::mlir::Type rhsRawType{};
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(&rhsRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawType = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatmulOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MatmulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::MatmulOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::MatmulOptimizedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MatmulOptimizedOpGenericAdaptorBase::MatmulOptimizedOpGenericAdaptorBase(MatmulOptimizedOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef MatmulOptimizedOpGenericAdaptorBase::getStrategy() {
  auto attr = getStrategyAttr();
  return attr.getValue();
}

::mlir::boas::NPUOptimizationAttr MatmulOptimizedOpGenericAdaptorBase::getConfig() {
  auto attr = getConfigAttr();
  return ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(attr);
}

} // namespace detail
MatmulOptimizedOpAdaptor::MatmulOptimizedOpAdaptor(MatmulOptimizedOp op) : MatmulOptimizedOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult MatmulOptimizedOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_config = getProperties().config; (void)tblgen_config;
  if (!tblgen_config) return emitError(loc, "'boas.matmul.optimized' op ""requires attribute 'config'");
  auto tblgen_strategy = getProperties().strategy; (void)tblgen_strategy;
  if (!tblgen_strategy) return emitError(loc, "'boas.matmul.optimized' op ""requires attribute 'strategy'");

  if (tblgen_strategy && !((::llvm::isa<::mlir::StringAttr>(tblgen_strategy))))
    return emitError(loc, "'boas.matmul.optimized' op ""attribute 'strategy' failed to satisfy constraint: string attribute");

  if (tblgen_config && !((::llvm::isa<::mlir::boas::NPUOptimizationAttr>(tblgen_config))))
    return emitError(loc, "'boas.matmul.optimized' op ""attribute 'config' failed to satisfy constraint: NPU\344\274\230\345\214\226\351\205\215\347\275\256\345\261\236\346\200\247");
  return ::mlir::success();
}

::llvm::LogicalResult MatmulOptimizedOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.config;
       auto attr = dict.get("config");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `config` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.strategy;
       auto attr = dict.get("strategy");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `strategy` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute MatmulOptimizedOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.config;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("config",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.strategy;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("strategy",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code MatmulOptimizedOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.config.getAsOpaquePointer()), 
    llvm::hash_value(prop.strategy.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> MatmulOptimizedOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "config")
      return prop.config;

    if (name == "strategy")
      return prop.strategy;
  return std::nullopt;
}

void MatmulOptimizedOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "config") {
       prop.config = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.config)>>(value);
       return;
    }

    if (name == "strategy") {
       prop.strategy = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.strategy)>>(value);
       return;
    }
}

void MatmulOptimizedOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.config) attrs.append("config", prop.config);

    if (prop.strategy) attrs.append("strategy", prop.strategy);
}

::llvm::LogicalResult MatmulOptimizedOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getConfigAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps2(attr, "config", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrategyAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(attr, "strategy", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult MatmulOptimizedOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.config)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.strategy)))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatmulOptimizedOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.config);
  writer.writeAttribute(prop.strategy);
}

::llvm::StringRef MatmulOptimizedOp::getStrategy() {
  auto attr = getStrategyAttr();
  return attr.getValue();
}

::mlir::boas::NPUOptimizationAttr MatmulOptimizedOp::getConfig() {
  auto attr = getConfigAttr();
  return ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(attr);
}

void MatmulOptimizedOp::setStrategy(::llvm::StringRef attrValue) {
  getProperties().strategy = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void MatmulOptimizedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::StringAttr strategy, ::mlir::boas::NPUOptimizationAttr config) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.getOrAddProperties<Properties>().strategy = strategy;
  odsState.getOrAddProperties<Properties>().config = config;
  odsState.addTypes(result);
}

void MatmulOptimizedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::StringAttr strategy, ::mlir::boas::NPUOptimizationAttr config) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.getOrAddProperties<Properties>().strategy = strategy;
  odsState.getOrAddProperties<Properties>().config = config;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatmulOptimizedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, ::llvm::StringRef strategy, ::mlir::boas::NPUOptimizationAttr config) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.getOrAddProperties<Properties>().strategy = odsBuilder.getStringAttr(strategy);
  odsState.getOrAddProperties<Properties>().config = config;
  odsState.addTypes(result);
}

void MatmulOptimizedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::llvm::StringRef strategy, ::mlir::boas::NPUOptimizationAttr config) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.getOrAddProperties<Properties>().strategy = odsBuilder.getStringAttr(strategy);
  odsState.getOrAddProperties<Properties>().config = config;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatmulOptimizedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<MatmulOptimizedOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult MatmulOptimizedOp::verifyInvariantsImpl() {
  auto tblgen_config = getProperties().config; (void)tblgen_config;
  if (!tblgen_config) return emitOpError("requires attribute 'config'");
  auto tblgen_strategy = getProperties().strategy; (void)tblgen_strategy;
  if (!tblgen_strategy) return emitOpError("requires attribute 'strategy'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(*this, tblgen_strategy, "strategy")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps2(*this, tblgen_config, "config")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult MatmulOptimizedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MatmulOptimizedOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(&lhsRawOperand, 1);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(&rhsRawOperand, 1);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::StringAttr strategyAttr;
  ::mlir::boas::NPUOptimizationAttr configAttr;
  ::mlir::Type lhsRawType{};
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(&lhsRawType, 1);
  ::mlir::Type rhsRawType{};
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(&rhsRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("strategy"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(strategyAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (strategyAttr) result.getOrAddProperties<MatmulOptimizedOp::Properties>().strategy = strategyAttr;
  if (parser.parseKeyword("config"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(configAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (configAttr) result.getOrAddProperties<MatmulOptimizedOp::Properties>().config = configAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawType = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatmulOptimizedOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ' << "strategy";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getStrategyAttr());
  _odsPrinter << ' ' << "config";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getConfigAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("strategy");
  elidedAttrs.push_back("config");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MatmulOptimizedOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::MatmulOptimizedOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::NPUKernelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
NPUKernelOpGenericAdaptorBase::NPUKernelOpGenericAdaptorBase(NPUKernelOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> NPUKernelOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::StringRef NPUKernelOpGenericAdaptorBase::getKernelName() {
  auto attr = getKernelNameAttr();
  return attr.getValue();
}

::mlir::boas::NPUOptimizationAttr NPUKernelOpGenericAdaptorBase::getConfig() {
  auto attr = getConfigAttr();
  return ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(attr);
}

::std::optional<uint64_t> NPUKernelOpGenericAdaptorBase::getGridSize() {
  auto attr = getGridSizeAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::std::optional<uint64_t> NPUKernelOpGenericAdaptorBase::getBlockSize() {
  auto attr = getBlockSizeAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
NPUKernelOpAdaptor::NPUKernelOpAdaptor(NPUKernelOp op) : NPUKernelOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult NPUKernelOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_block_size = getProperties().block_size; (void)tblgen_block_size;
  auto tblgen_config = getProperties().config; (void)tblgen_config;
  if (!tblgen_config) return emitError(loc, "'boas.npu.kernel' op ""requires attribute 'config'");
  auto tblgen_grid_size = getProperties().grid_size; (void)tblgen_grid_size;
  auto tblgen_kernel_name = getProperties().kernel_name; (void)tblgen_kernel_name;
  if (!tblgen_kernel_name) return emitError(loc, "'boas.npu.kernel' op ""requires attribute 'kernel_name'");

  if (tblgen_kernel_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_kernel_name))))
    return emitError(loc, "'boas.npu.kernel' op ""attribute 'kernel_name' failed to satisfy constraint: string attribute");

  if (tblgen_config && !((::llvm::isa<::mlir::boas::NPUOptimizationAttr>(tblgen_config))))
    return emitError(loc, "'boas.npu.kernel' op ""attribute 'config' failed to satisfy constraint: NPU\344\274\230\345\214\226\351\205\215\347\275\256\345\261\236\346\200\247");

  if (tblgen_grid_size && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_grid_size))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_grid_size).getType().isSignlessInteger(64)))))
    return emitError(loc, "'boas.npu.kernel' op ""attribute 'grid_size' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_block_size && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_block_size))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_block_size).getType().isSignlessInteger(64)))))
    return emitError(loc, "'boas.npu.kernel' op ""attribute 'block_size' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> NPUKernelOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange NPUKernelOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> NPUKernelOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult NPUKernelOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.block_size;
       auto attr = dict.get("block_size");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `block_size` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.config;
       auto attr = dict.get("config");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `config` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.grid_size;
       auto attr = dict.get("grid_size");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `grid_size` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.kernel_name;
       auto attr = dict.get("kernel_name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `kernel_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute NPUKernelOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.block_size;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("block_size",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.config;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("config",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.grid_size;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("grid_size",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.kernel_name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("kernel_name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code NPUKernelOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.block_size.getAsOpaquePointer()), 
    llvm::hash_value(prop.config.getAsOpaquePointer()), 
    llvm::hash_value(prop.grid_size.getAsOpaquePointer()), 
    llvm::hash_value(prop.kernel_name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> NPUKernelOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "block_size")
      return prop.block_size;

    if (name == "config")
      return prop.config;

    if (name == "grid_size")
      return prop.grid_size;

    if (name == "kernel_name")
      return prop.kernel_name;
  return std::nullopt;
}

void NPUKernelOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "block_size") {
       prop.block_size = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.block_size)>>(value);
       return;
    }

    if (name == "config") {
       prop.config = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.config)>>(value);
       return;
    }

    if (name == "grid_size") {
       prop.grid_size = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.grid_size)>>(value);
       return;
    }

    if (name == "kernel_name") {
       prop.kernel_name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.kernel_name)>>(value);
       return;
    }
}

void NPUKernelOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.block_size) attrs.append("block_size", prop.block_size);

    if (prop.config) attrs.append("config", prop.config);

    if (prop.grid_size) attrs.append("grid_size", prop.grid_size);

    if (prop.kernel_name) attrs.append("kernel_name", prop.kernel_name);
}

::llvm::LogicalResult NPUKernelOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBlockSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps3(attr, "block_size", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getConfigAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps2(attr, "config", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getGridSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps3(attr, "grid_size", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getKernelNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(attr, "kernel_name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult NPUKernelOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.block_size)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.config)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.grid_size)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.kernel_name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void NPUKernelOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.block_size);
  writer.writeAttribute(prop.config);

  writer.writeOptionalAttribute(prop.grid_size);
  writer.writeAttribute(prop.kernel_name);
}

::llvm::StringRef NPUKernelOp::getKernelName() {
  auto attr = getKernelNameAttr();
  return attr.getValue();
}

::mlir::boas::NPUOptimizationAttr NPUKernelOp::getConfig() {
  auto attr = getConfigAttr();
  return ::llvm::cast<::mlir::boas::NPUOptimizationAttr>(attr);
}

::std::optional<uint64_t> NPUKernelOp::getGridSize() {
  auto attr = getGridSizeAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::std::optional<uint64_t> NPUKernelOp::getBlockSize() {
  auto attr = getBlockSizeAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void NPUKernelOp::setKernelName(::llvm::StringRef attrValue) {
  getProperties().kernel_name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void NPUKernelOp::setGridSize(::std::optional<uint64_t> attrValue) {
    auto &odsProp = getProperties().grid_size;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue);
    else
      odsProp = nullptr;
}

void NPUKernelOp::setBlockSize(::std::optional<uint64_t> attrValue) {
    auto &odsProp = getProperties().block_size;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue);
    else
      odsProp = nullptr;
}

void NPUKernelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange operands, ::mlir::StringAttr kernel_name, ::mlir::boas::NPUOptimizationAttr config, /*optional*/::mlir::IntegerAttr grid_size, /*optional*/::mlir::IntegerAttr block_size) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().kernel_name = kernel_name;
  odsState.getOrAddProperties<Properties>().config = config;
  if (grid_size) {
    odsState.getOrAddProperties<Properties>().grid_size = grid_size;
  }
  if (block_size) {
    odsState.getOrAddProperties<Properties>().block_size = block_size;
  }
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void NPUKernelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange operands, ::llvm::StringRef kernel_name, ::mlir::boas::NPUOptimizationAttr config, /*optional*/::mlir::IntegerAttr grid_size, /*optional*/::mlir::IntegerAttr block_size) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().kernel_name = odsBuilder.getStringAttr(kernel_name);
  odsState.getOrAddProperties<Properties>().config = config;
  if (grid_size) {
    odsState.getOrAddProperties<Properties>().grid_size = grid_size;
  }
  if (block_size) {
    odsState.getOrAddProperties<Properties>().block_size = block_size;
  }
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void NPUKernelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<NPUKernelOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult NPUKernelOp::verifyInvariantsImpl() {
  auto tblgen_block_size = getProperties().block_size; (void)tblgen_block_size;
  auto tblgen_config = getProperties().config; (void)tblgen_config;
  if (!tblgen_config) return emitOpError("requires attribute 'config'");
  auto tblgen_grid_size = getProperties().grid_size; (void)tblgen_grid_size;
  auto tblgen_kernel_name = getProperties().kernel_name; (void)tblgen_kernel_name;
  if (!tblgen_kernel_name) return emitOpError("requires attribute 'kernel_name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(*this, tblgen_kernel_name, "kernel_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps2(*this, tblgen_config, "config")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps3(*this, tblgen_grid_size, "grid_size")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps3(*this, tblgen_block_size, "block_size")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_BoasOps1(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::llvm::LogicalResult NPUKernelOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult NPUKernelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::StringAttr kernel_nameAttr;
  ::mlir::boas::NPUOptimizationAttr configAttr;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> operandsTypes;
  ::llvm::ArrayRef<::mlir::Type> resultsTypes;

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (parser.parseKeyword("kernel"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(kernel_nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (kernel_nameAttr) result.getOrAddProperties<NPUKernelOp::Properties>().kernel_name = kernel_nameAttr;
  if (parser.parseKeyword("config"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(configAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (configAttr) result.getOrAddProperties<NPUKernelOp::Properties>().config = configAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType operands__results_functionType;
  if (parser.parseType(operands__results_functionType))
    return ::mlir::failure();
  operandsTypes = operands__results_functionType.getInputs();
  resultsTypes = operands__results_functionType.getResults();
  result.addRegion(std::move(bodyRegion));
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void NPUKernelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperands();
  _odsPrinter << ' ' << "kernel";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getKernelNameAttr());
  _odsPrinter << ' ' << "config";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getConfigAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("kernel_name");
  elidedAttrs.push_back("config");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBody());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperands().getTypes(), getResults().getTypes());
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::NPUKernelOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::NPULaunchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
NPULaunchOpAdaptor::NPULaunchOpAdaptor(NPULaunchOp op) : NPULaunchOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult NPULaunchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void NPULaunchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value gridX, ::mlir::Value gridY, ::mlir::Value gridZ, ::mlir::Value blockX, ::mlir::Value blockY, ::mlir::Value blockZ) {
  odsState.addOperands(gridX);
  odsState.addOperands(gridY);
  odsState.addOperands(gridZ);
  odsState.addOperands(blockX);
  odsState.addOperands(blockY);
  odsState.addOperands(blockZ);
  (void)odsState.addRegion();
}

void NPULaunchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value gridX, ::mlir::Value gridY, ::mlir::Value gridZ, ::mlir::Value blockX, ::mlir::Value blockY, ::mlir::Value blockZ) {
  odsState.addOperands(gridX);
  odsState.addOperands(gridY);
  odsState.addOperands(gridZ);
  odsState.addOperands(blockX);
  odsState.addOperands(blockY);
  odsState.addOperands(blockZ);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NPULaunchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult NPULaunchOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_BoasOps1(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::llvm::LogicalResult NPULaunchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult NPULaunchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand gridXRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> gridXOperands(&gridXRawOperand, 1);  ::llvm::SMLoc gridXOperandsLoc;
  (void)gridXOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand gridYRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> gridYOperands(&gridYRawOperand, 1);  ::llvm::SMLoc gridYOperandsLoc;
  (void)gridYOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand gridZRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> gridZOperands(&gridZRawOperand, 1);  ::llvm::SMLoc gridZOperandsLoc;
  (void)gridZOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand blockXRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> blockXOperands(&blockXRawOperand, 1);  ::llvm::SMLoc blockXOperandsLoc;
  (void)blockXOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand blockYRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> blockYOperands(&blockYRawOperand, 1);  ::llvm::SMLoc blockYOperandsLoc;
  (void)blockYOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand blockZRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> blockZOperands(&blockZRawOperand, 1);  ::llvm::SMLoc blockZOperandsLoc;
  (void)blockZOperandsLoc;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();
  if (parser.parseKeyword("grid"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  gridXOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(gridXRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  gridYOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(gridYRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  gridZOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(gridZRawOperand))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseKeyword("block"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  blockXOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(blockXRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  blockYOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(blockYRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  blockZOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(blockZRawOperand))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return ::mlir::failure();
  }

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(gridXOperands, odsBuildableType0, gridXOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gridYOperands, odsBuildableType0, gridYOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(gridZOperands, odsBuildableType0, gridZOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(blockXOperands, odsBuildableType0, blockXOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(blockYOperands, odsBuildableType0, blockYOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(blockZOperands, odsBuildableType0, blockZOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void NPULaunchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "grid";
  _odsPrinter << "(";
  _odsPrinter << getGridX();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getGridY();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getGridZ();
  _odsPrinter << ")";
  _odsPrinter << ' ' << "block";
  _odsPrinter << "(";
  _odsPrinter << getBlockX();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getBlockY();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getBlockZ();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBody());
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::NPULaunchOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::PrintOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
PrintOpAdaptor::PrintOpAdaptor(PrintOp op) : PrintOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult PrintOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult PrintOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult PrintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::PrintOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::ShapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ShapeOpAdaptor::ShapeOpAdaptor(ShapeOp op) : ShapeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ShapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ShapeOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "rows");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "cols");
}

void ShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type rows, ::mlir::Type cols, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(rows);
  odsState.addTypes(cols);
}

void ShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ShapeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

void ShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ShapeOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 2u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::llvm::LogicalResult ShapeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ShapeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult ShapeOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(2);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  ::mlir::Type odsInferredType1 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  inferredReturnTypes[1] = odsInferredType1;
  return ::mlir::success();
}

::mlir::ParseResult ShapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);
  ::mlir::Type rowsRawType{};
  ::llvm::ArrayRef<::mlir::Type> rowsTypes(&rowsRawType, 1);
  ::mlir::Type colsRawType{};
  ::llvm::ArrayRef<::mlir::Type> colsTypes(&colsRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::IndexType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rowsRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::IndexType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    colsRawType = type;
  }
  result.addTypes(rowsTypes);
  result.addTypes(colsTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getRows().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IndexType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getCols().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IndexType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ShapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::ShapeOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::TensorCreateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TensorCreateOpGenericAdaptorBase::TensorCreateOpGenericAdaptorBase(TensorCreateOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef TensorCreateOpGenericAdaptorBase::getDevice() {
  auto attr = getDeviceAttr();
  return attr.getValue();
}

} // namespace detail
TensorCreateOpAdaptor::TensorCreateOpAdaptor(TensorCreateOp op) : TensorCreateOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TensorCreateOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_device = getProperties().device; (void)tblgen_device;
  if (!tblgen_device) return emitError(loc, "'boas.tensor.create' op ""requires attribute 'device'");

  if (tblgen_device && !((::llvm::isa<::mlir::StringAttr>(tblgen_device))))
    return emitError(loc, "'boas.tensor.create' op ""attribute 'device' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TensorCreateOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.device;
       auto attr = dict.get("device");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `device` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TensorCreateOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.device;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("device",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TensorCreateOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.device.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TensorCreateOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "device")
      return prop.device;
  return std::nullopt;
}

void TensorCreateOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "device") {
       prop.device = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.device)>>(value);
       return;
    }
}

void TensorCreateOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.device) attrs.append("device", prop.device);
}

::llvm::LogicalResult TensorCreateOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getDeviceAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(attr, "device", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TensorCreateOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.device)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorCreateOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.device);
}

::llvm::StringRef TensorCreateOp::getDevice() {
  auto attr = getDeviceAttr();
  return attr.getValue();
}

void TensorCreateOp::setDevice(::llvm::StringRef attrValue) {
  getProperties().device = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void TensorCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::mlir::StringAttr device) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.addOperands(values);
  odsState.getOrAddProperties<Properties>().device = device;
  odsState.addTypes(result);
}

void TensorCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::mlir::StringAttr device) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.addOperands(values);
  odsState.getOrAddProperties<Properties>().device = device;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(TensorCreateOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

void TensorCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::mlir::StringAttr device) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.addOperands(values);
  odsState.getOrAddProperties<Properties>().device = device;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::llvm::StringRef device) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.addOperands(values);
  odsState.getOrAddProperties<Properties>().device = odsBuilder.getStringAttr(device);
  odsState.addTypes(result);
}

void TensorCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::llvm::StringRef device) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.addOperands(values);
  odsState.getOrAddProperties<Properties>().device = odsBuilder.getStringAttr(device);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(TensorCreateOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

void TensorCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value rows, ::mlir::Value cols, ::mlir::Value values, ::llvm::StringRef device) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.addOperands(values);
  odsState.getOrAddProperties<Properties>().device = odsBuilder.getStringAttr(device);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorCreateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TensorCreateOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void TensorCreateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TensorCreateOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(TensorCreateOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::llvm::LogicalResult TensorCreateOp::verifyInvariantsImpl() {
  auto tblgen_device = getProperties().device; (void)tblgen_device;
  if (!tblgen_device) return emitOpError("requires attribute 'device'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(*this, tblgen_device, "device")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(Boas_TensorType::get(this->getOperation()->getContext(), (*this->getODSOperands(0).begin()).getValue(), (*this->getODSOperands(1).begin()).getValue(), (*this->getODSOperands(2).begin()).getType().cast<ShapedType>().getElementType(), getDevice().getValue()), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches element type");
  return ::mlir::success();
}

::llvm::LogicalResult TensorCreateOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult TensorCreateOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  if (operands.size() <= 2)
    return ::mlir::failure();
  ::mlir::Type odsInferredType0 = Boas_TensorType::get(context, $rows<no-subst-found>.getValue(), $cols<no-subst-found>.getValue(), $values<no-subst-found>.getType().cast<ShapedType>().getElementType(), $device<no-subst-found>.getValue());
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult TensorCreateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand rowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rowsOperands(&rowsRawOperand, 1);  ::llvm::SMLoc rowsOperandsLoc;
  (void)rowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand colsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> colsOperands(&colsRawOperand, 1);  ::llvm::SMLoc colsOperandsLoc;
  (void)colsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand valuesRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valuesOperands(&valuesRawOperand, 1);  ::llvm::SMLoc valuesOperandsLoc;
  (void)valuesOperandsLoc;
  ::mlir::Type valuesRawType{};
  ::llvm::ArrayRef<::mlir::Type> valuesTypes(&valuesRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);
  if (parser.parseLParen())
    return ::mlir::failure();

  rowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rowsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  colsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(colsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valuesRawOperand))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valuesRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(rowsOperands, odsBuildableType0, rowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(colsOperands, odsBuildableType0, colsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valuesOperands, valuesTypes, valuesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorCreateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getRows();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getCols();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getValues();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValues().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void TensorCreateOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::TensorCreateOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::TensorRandomOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TensorRandomOpGenericAdaptorBase::TensorRandomOpGenericAdaptorBase(TensorRandomOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef TensorRandomOpGenericAdaptorBase::getDevice() {
  auto attr = getDeviceAttr();
  return attr.getValue();
}

::std::optional< ::llvm::APFloat > TensorRandomOpGenericAdaptorBase::getSeed() {
  auto attr = getSeedAttr();
  return attr ? ::std::optional< ::llvm::APFloat >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
TensorRandomOpAdaptor::TensorRandomOpAdaptor(TensorRandomOp op) : TensorRandomOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TensorRandomOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_device = getProperties().device; (void)tblgen_device;
  if (!tblgen_device) return emitError(loc, "'boas.tensor.random' op ""requires attribute 'device'");
  auto tblgen_seed = getProperties().seed; (void)tblgen_seed;

  if (tblgen_device && !((::llvm::isa<::mlir::StringAttr>(tblgen_device))))
    return emitError(loc, "'boas.tensor.random' op ""attribute 'device' failed to satisfy constraint: string attribute");

  if (tblgen_seed && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_seed))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_seed).getType().isF64()))))
    return emitError(loc, "'boas.tensor.random' op ""attribute 'seed' failed to satisfy constraint: 64-bit float attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TensorRandomOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.device;
       auto attr = dict.get("device");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `device` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.seed;
       auto attr = dict.get("seed");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `seed` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TensorRandomOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.device;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("device",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.seed;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("seed",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TensorRandomOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.device.getAsOpaquePointer()), 
    llvm::hash_value(prop.seed.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TensorRandomOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "device")
      return prop.device;

    if (name == "seed")
      return prop.seed;
  return std::nullopt;
}

void TensorRandomOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "device") {
       prop.device = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.device)>>(value);
       return;
    }

    if (name == "seed") {
       prop.seed = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.seed)>>(value);
       return;
    }
}

void TensorRandomOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.device) attrs.append("device", prop.device);

    if (prop.seed) attrs.append("seed", prop.seed);
}

::llvm::LogicalResult TensorRandomOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getDeviceAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(attr, "device", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSeedAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps4(attr, "seed", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TensorRandomOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.device)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.seed)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorRandomOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.device);

  writer.writeOptionalAttribute(prop.seed);
}

::llvm::StringRef TensorRandomOp::getDevice() {
  auto attr = getDeviceAttr();
  return attr.getValue();
}

::std::optional< ::llvm::APFloat > TensorRandomOp::getSeed() {
  auto attr = getSeedAttr();
  return attr ? ::std::optional< ::llvm::APFloat >(attr.getValue()) : (::std::nullopt);
}

void TensorRandomOp::setDevice(::llvm::StringRef attrValue) {
  getProperties().device = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void TensorRandomOp::setSeed(::std::optional<::llvm::APFloat> attrValue) {
    auto &odsProp = getProperties().seed;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF64Type(), *attrValue);
    else
      odsProp = nullptr;
}

void TensorRandomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value rows, ::mlir::Value cols, ::mlir::StringAttr device, /*optional*/::mlir::FloatAttr seed) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.getOrAddProperties<Properties>().device = device;
  if (seed) {
    odsState.getOrAddProperties<Properties>().seed = seed;
  }
  odsState.addTypes(result);
}

void TensorRandomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value rows, ::mlir::Value cols, ::mlir::StringAttr device, /*optional*/::mlir::FloatAttr seed) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.getOrAddProperties<Properties>().device = device;
  if (seed) {
    odsState.getOrAddProperties<Properties>().seed = seed;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorRandomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value rows, ::mlir::Value cols, ::llvm::StringRef device, /*optional*/::mlir::FloatAttr seed) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.getOrAddProperties<Properties>().device = odsBuilder.getStringAttr(device);
  if (seed) {
    odsState.getOrAddProperties<Properties>().seed = seed;
  }
  odsState.addTypes(result);
}

void TensorRandomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value rows, ::mlir::Value cols, ::llvm::StringRef device, /*optional*/::mlir::FloatAttr seed) {
  odsState.addOperands(rows);
  odsState.addOperands(cols);
  odsState.getOrAddProperties<Properties>().device = odsBuilder.getStringAttr(device);
  if (seed) {
    odsState.getOrAddProperties<Properties>().seed = seed;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorRandomOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TensorRandomOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult TensorRandomOp::verifyInvariantsImpl() {
  auto tblgen_device = getProperties().device; (void)tblgen_device;
  if (!tblgen_device) return emitOpError("requires attribute 'device'");
  auto tblgen_seed = getProperties().seed; (void)tblgen_seed;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(*this, tblgen_device, "device")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps4(*this, tblgen_seed, "seed")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TensorRandomOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TensorRandomOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand rowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rowsOperands(&rowsRawOperand, 1);  ::llvm::SMLoc rowsOperandsLoc;
  (void)rowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand colsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> colsOperands(&colsRawOperand, 1);  ::llvm::SMLoc colsOperandsLoc;
  (void)colsOperandsLoc;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);
  if (parser.parseLParen())
    return ::mlir::failure();

  rowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rowsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  colsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(colsRawOperand))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(rowsOperands, odsBuildableType0, rowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(colsOperands, odsBuildableType0, colsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorRandomOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getRows();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getCols();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void TensorRandomOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::TensorRandomOp)

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// ::mlir::boas::ToDeviceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ToDeviceOpGenericAdaptorBase::ToDeviceOpGenericAdaptorBase(ToDeviceOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef ToDeviceOpGenericAdaptorBase::getTargetDevice() {
  auto attr = getTargetDeviceAttr();
  return attr.getValue();
}

} // namespace detail
ToDeviceOpAdaptor::ToDeviceOpAdaptor(ToDeviceOp op) : ToDeviceOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ToDeviceOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_target_device = getProperties().target_device; (void)tblgen_target_device;
  if (!tblgen_target_device) return emitError(loc, "'boas.to_device' op ""requires attribute 'target_device'");

  if (tblgen_target_device && !((::llvm::isa<::mlir::StringAttr>(tblgen_target_device))))
    return emitError(loc, "'boas.to_device' op ""attribute 'target_device' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ToDeviceOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.target_device;
       auto attr = dict.get("target_device");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `target_device` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ToDeviceOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.target_device;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("target_device",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ToDeviceOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.target_device.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ToDeviceOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "target_device")
      return prop.target_device;
  return std::nullopt;
}

void ToDeviceOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "target_device") {
       prop.target_device = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.target_device)>>(value);
       return;
    }
}

void ToDeviceOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.target_device) attrs.append("target_device", prop.target_device);
}

::llvm::LogicalResult ToDeviceOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTargetDeviceAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(attr, "target_device", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ToDeviceOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.target_device)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ToDeviceOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.target_device);
}

::llvm::StringRef ToDeviceOp::getTargetDevice() {
  auto attr = getTargetDeviceAttr();
  return attr.getValue();
}

void ToDeviceOp::setTargetDevice(::llvm::StringRef attrValue) {
  getProperties().target_device = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void ToDeviceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::StringAttr target_device) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().target_device = target_device;
  odsState.addTypes(result);
}

void ToDeviceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::StringAttr target_device) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().target_device = target_device;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToDeviceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::llvm::StringRef target_device) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().target_device = odsBuilder.getStringAttr(target_device);
  odsState.addTypes(result);
}

void ToDeviceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::StringRef target_device) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().target_device = odsBuilder.getStringAttr(target_device);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToDeviceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ToDeviceOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult ToDeviceOp::verifyInvariantsImpl() {
  auto tblgen_target_device = getProperties().target_device; (void)tblgen_target_device;
  if (!tblgen_target_device) return emitOpError("requires attribute 'target_device'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BoasOps1(*this, tblgen_target_device, "target_device")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BoasOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ToDeviceOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ToDeviceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::StringAttr target_deviceAttr;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(target_deviceAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (target_deviceAttr) result.getOrAddProperties<ToDeviceOp::Properties>().target_device = target_deviceAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::boas::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ToDeviceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTargetDeviceAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("target_device");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::boas::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ToDeviceOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace boas
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::boas::ToDeviceOp)


#endif  // GET_OP_CLASSES

