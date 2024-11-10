#ifndef BOAS_MLIR_CONVERSION_BOAS_TO_LLVM_H
#define BOAS_MLIR_CONVERSION_BOAS_TO_LLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace boas {
namespace mlir {

std::unique_ptr<::mlir::Pass> createConvertBoasToLLVMPass();

class BoasToLLVMTypeConverter : public ::mlir::LLVMTypeConverter {
public:
    explicit BoasToLLVMTypeConverter(::mlir::MLIRContext *ctx);
};

} // namespace mlir
} // namespace boas

#endif
