#ifndef BOAS_MLIR_CONVERSION_BOAS_TO_LLVM_H
#define BOAS_MLIR_CONVERSION_BOAS_TO_LLVM_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace boas {
namespace mlir {

class BoasToLLVMTypeConverter : public ::mlir::TypeConverter {
public:
    explicit BoasToLLVMTypeConverter(::mlir::MLIRContext *ctx);
};

void populateBoasToLLVMConversionPatterns(BoasToLLVMTypeConverter &typeConverter,
                                         ::mlir::RewritePatternSet &patterns);

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createConvertBoasToLLVMPass();

} // namespace mlir
} // namespace boas

#endif // BOAS_MLIR_CONVERSION_BOAS_TO_LLVM_H