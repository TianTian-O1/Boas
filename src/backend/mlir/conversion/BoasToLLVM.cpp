#include "boas/backend/mlir/BoasToLLVM.h"
#include "boas/backend/mlir/ListOps.h"
#include "boas/backend/mlir/NumberOps.h"
#include "boas/backend/mlir/BoasDialect.h"
#include "boas/backend/mlir/PrintOps.h"
#include "boas/backend/mlir/TensorOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace boas {
namespace mlir {

BoasToLLVMTypeConverter::BoasToLLVMTypeConverter(::mlir::MLIRContext *context)
    : ::mlir::TypeConverter() {
    // Add standard type conversions
    addConversion([](::mlir::Type type) { return type; });
    
    // Convert list type to LLVM pointer type
    addConversion([context](::mlir::Type type) -> ::mlir::Type {
        if (type.isa<::mlir::UnrankedTensorType>()) {
            return ::mlir::LLVM::LLVMPointerType::get(context);
        }
        return type;
    });

    // Add materializations
    addSourceMaterialization([&](::mlir::OpBuilder &builder, ::mlir::Type resultType,
                               ::mlir::ValueRange inputs, ::mlir::Location loc) -> std::optional<::mlir::Value> {
        if (inputs.size() != 1)
            return std::nullopt;
        return inputs[0];
    });
}

struct ListCreateOpLowering : public ::mlir::ConversionPattern {
    explicit ListCreateOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ListCreateOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        llvm::outs() << "ListCreateOp: Creating new list\n";
        
        // Create struct type for list
        auto elementTy = ::mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
        auto structTy = ::mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {elementTy, rewriter.getI64Type()}  // data pointer and size
        );
        
        // Get malloc function
        auto moduleOp = op->getParentOfType<::mlir::ModuleOp>();
        auto mallocRef = ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "malloc");
        
        // Allocate memory for list struct
        auto mallocSize = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), 16);  // Size for pointer + size field
            
        llvm::outs() << "ListCreateOp: Allocating " << 16 << " bytes\n";
        
        auto allocated = rewriter.create<::mlir::LLVM::CallOp>(
            loc,
            ::mlir::TypeRange{::mlir::LLVM::LLVMPointerType::get(rewriter.getContext())},
            mallocRef,
            ::mlir::ValueRange{mallocSize}).getResult();

        // Initialize fields
        auto zero = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), 0);
        auto nullPtr = rewriter.create<::mlir::LLVM::ZeroOp>(loc, elementTy);
        
        llvm::outs() << "ListCreateOp: Initializing list structure\n";
        
        rewriter.replaceOp(op, allocated);
        return ::mlir::success();
    }
};

struct ListAppendOpLowering : public ::mlir::ConversionPattern {
    explicit ListAppendOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ListAppendOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto appendOp = ::mlir::cast<ListAppendOp>(op);
        auto loc = op->getLoc();
        
        // Get list and element to append
        auto list = appendOp->getOperand(0);
        auto element = appendOp->getOperand(1);
        
        // Create realloc function declaration if it doesn't exist
        auto moduleOp = op->getParentOfType<::mlir::ModuleOp>();
        auto reallocFunc = moduleOp.lookupSymbol<::mlir::LLVM::LLVMFuncOp>("realloc");
        if (!reallocFunc) {
            auto reallocType = ::mlir::LLVM::LLVMFunctionType::get(
                ::mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
                {::mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), 
                 rewriter.getI64Type()});
            reallocFunc = rewriter.create<::mlir::LLVM::LLVMFuncOp>(
                loc, "realloc", reallocType);
        }
        
        // Calculate new size (current size + 1)
        
        return ::mlir::success();
    }
};

struct NumberConstantOpLowering : public ::mlir::ConversionPattern {
    explicit NumberConstantOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(NumberConstantOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto constOp = ::mlir::cast<NumberConstantOp>(op);
        auto value = rewriter.create<::mlir::LLVM::ConstantOp>(
            op->getLoc(),
            rewriter.getI64Type(),
            constOp.getValue()
        );
        rewriter.replaceOp(op, value);
        return ::mlir::success();
    }
};

struct PrintOpLowering : public ::mlir::ConversionPattern {
    explicit PrintOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(PrintOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto printOp = ::mlir::cast<PrintOp>(op);
        auto loc = op->getLoc();
        
        // 调用print方法进行打印
        if (printOp.print().failed()) {
            return ::mlir::failure();
        }
        
        // 替换操作
        rewriter.replaceOp(op, printOp.getOperand());
        return ::mlir::success();
    }
};

struct ListSliceOpLowering : public ::mlir::ConversionPattern {
    explicit ListSliceOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ListSliceOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto sliceOp = ::llvm::dyn_cast<ListSliceOp>(op);
        if (!sliceOp)
            return ::mlir::failure();
            
        // TODO: Implement slice operation lowering
        
        return ::mlir::success();
    }
};

struct ListNestedCreateOpLowering : public ::mlir::ConversionPattern {
    explicit ListNestedCreateOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ListNestedCreateOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        
        // Create struct type for nested list
        auto elementTy = ::mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
        auto structTy = ::mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {elementTy, rewriter.getI64Type()}  // data pointer and size
        );
            
        // Get malloc function
        auto moduleOp = op->getParentOfType<::mlir::ModuleOp>();
        auto mallocRef = ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "malloc");
        
        // Allocate memory for nested list
        auto mallocSize = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), 
            16);  // Size for pointer (8 bytes) + size field (8 bytes)

        // Create malloc call
        auto allocated = rewriter.create<::mlir::LLVM::CallOp>(
            loc,
            ::mlir::TypeRange{::mlir::LLVM::LLVMPointerType::get(rewriter.getContext())},
            mallocRef,
            ::mlir::ValueRange{mallocSize}).getResult();
            
        // Initialize the struct fields to zero
        auto zero = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), 0);
        auto nullPtr = rewriter.create<::mlir::LLVM::ZeroOp>(
            loc, elementTy);
            
        // Store initial values
        auto dataPtr = rewriter.create<::mlir::LLVM::GEPOp>(
            loc,
            elementTy,
            structTy,
            allocated,
            ::mlir::ValueRange{zero, zero},
            false);
        rewriter.create<::mlir::LLVM::StoreOp>(loc, nullPtr, dataPtr);
        
        auto sizePtr = rewriter.create<::mlir::LLVM::GEPOp>(
            loc,
            rewriter.getI64Type(),
            structTy,
            allocated,
            ::mlir::ValueRange{zero, rewriter.create<::mlir::LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), 1)},
            false);
        rewriter.create<::mlir::LLVM::StoreOp>(loc, zero, sizePtr);
            
        rewriter.replaceOp(op, allocated);
        return ::mlir::success();
    }
};

struct ListGetOpLowering : public ::mlir::ConversionPattern {
    explicit ListGetOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ListGetOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto getOp = ::mlir::cast<ListGetOp>(op);
        auto loc = op->getLoc();
        
        // Get list and index
        auto list = getOp.getList();
        auto index = getOp.getIndex();
        
        // Get list struct type
        auto elementTy = ::mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
        auto structTy = ::mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {elementTy, rewriter.getI64Type()}  // data pointer and size
        );
        
        // Create GEP operation to access data pointer
        auto zero = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), 0);
            
        // Access data pointer field in struct
        auto dataPtr = rewriter.create<::mlir::LLVM::GEPOp>(
            loc,
            elementTy,
            structTy,
            list,
            ::mlir::ValueRange{zero, zero},
            false);  // inbounds flag
            
        // Load data pointer
        auto data = rewriter.create<::mlir::LLVM::LoadOp>(
            loc,
            elementTy,
            dataPtr,
            0);
            
        // Use index to access element
        auto element = rewriter.create<::mlir::LLVM::GEPOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64Type(),
            data,
            ::mlir::ValueRange{index},
            false);
            
        // Load element value
        auto result = rewriter.create<::mlir::LLVM::LoadOp>(
            loc,
            rewriter.getI64Type(),
            element,
            0);
            
        rewriter.replaceOp(op, result.getResult());
        return ::mlir::success();
    }
};

struct TensorCreateOpLowering : public ::mlir::ConversionPattern {
    explicit TensorCreateOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(boas::mlir::TensorCreateOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto createOp = ::llvm::cast<boas::mlir::TensorCreateOp>(op);
        auto loc = op->getLoc();
        
        // Get tensor shape
        auto shape = createOp.getShape();
        
        // Allocate memory
        auto resultSize = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), shape[0] * shape[1] * sizeof(double));
            
        auto mallocRef = ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "malloc");
        auto allocated = rewriter.create<::mlir::LLVM::CallOp>(
            loc,
            ::mlir::TypeRange{::mlir::LLVM::LLVMPointerType::get(rewriter.getContext())},
            mallocRef,
            ::mlir::ValueRange{resultSize}).getResult();
            
        rewriter.replaceOp(op, allocated);
        return ::mlir::success();
    }
};

struct TensorMatMulOpLowering : public ::mlir::ConversionPattern {
    explicit TensorMatMulOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(boas::mlir::TensorMatMulOp::getOperationName(), 1, ctx) {}
        
    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ::llvm::ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto matmulOp = ::llvm::cast<boas::mlir::TensorMatMulOp>(op);
        auto loc = op->getLoc();
        
        // Get input matrix dimensions
        auto lhsType = matmulOp.getOperand(0).getType().dyn_cast<::mlir::ShapedType>();
        auto rhsType = matmulOp.getOperand(1).getType().dyn_cast<::mlir::ShapedType>();
        
        if (!lhsType || !rhsType) {
            return ::mlir::failure();
        }
        
        auto lhsShape = lhsType.getShape();
        auto rhsShape = rhsType.getShape();
        
        if (lhsShape.size() != 2 || rhsShape.size() != 2) {
            return ::mlir::failure();
        }
        
        auto m = lhsShape[0];
        auto k = lhsShape[1];
        auto n = rhsShape[1];
        
        // Allocate result matrix
        auto resultSize = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), m * n * sizeof(double));
            
        auto mallocRef = ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "malloc");
        auto result = rewriter.create<::mlir::LLVM::CallOp>(
            loc,
            ::mlir::TypeRange{::mlir::LLVM::LLVMPointerType::get(rewriter.getContext())},
            mallocRef,
            ::mlir::ValueRange{resultSize}).getResult();

        // Initialize result matrix with zeros
        auto zero = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getF64Type(), 0.0);

        // Create loops for matrix multiplication
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                auto idx_res = i * n + j;
                auto resPtr = rewriter.create<::mlir::LLVM::GEPOp>(
                    loc,
                    ::mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
                    rewriter.getF64Type(),
                    result,
                    ::mlir::ValueRange{rewriter.create<::mlir::LLVM::ConstantOp>(
                        loc, rewriter.getI64Type(), idx_res)},
                    false);
                        
                rewriter.create<::mlir::LLVM::StoreOp>(loc, zero, resPtr);
                
                for (int64_t p = 0; p < k; p++) {
                    auto idx1 = i * k + p;
                    auto idx2 = p * n + j;
                    
                    auto lhsPtr = rewriter.create<::mlir::LLVM::GEPOp>(
                        loc,
                        ::mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
                        rewriter.getF64Type(),
                        matmulOp.getOperand(0),
                        ::mlir::ValueRange{rewriter.create<::mlir::LLVM::ConstantOp>(
                            loc, rewriter.getI64Type(), idx1)},
                        false);
                            
                    auto rhsPtr = rewriter.create<::mlir::LLVM::GEPOp>(
                        loc,
                        ::mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
                        rewriter.getF64Type(),
                        matmulOp.getOperand(1),
                        ::mlir::ValueRange{rewriter.create<::mlir::LLVM::ConstantOp>(
                            loc, rewriter.getI64Type(), idx2)},
                        false);
                            
                    auto lhsVal = rewriter.create<::mlir::LLVM::LoadOp>(loc, rewriter.getF64Type(), lhsPtr);
                    auto rhsVal = rewriter.create<::mlir::LLVM::LoadOp>(loc, rewriter.getF64Type(), rhsPtr);
                    auto prod = rewriter.create<::mlir::LLVM::FMulOp>(loc, lhsVal, rhsVal);
                    
                    auto currVal = rewriter.create<::mlir::LLVM::LoadOp>(loc, rewriter.getF64Type(), resPtr);
                    auto sum = rewriter.create<::mlir::LLVM::FAddOp>(loc, currVal, prod);
                    rewriter.create<::mlir::LLVM::StoreOp>(loc, sum, resPtr);
                }
            }
        }
        
        rewriter.replaceOp(op, result);
        return ::mlir::success();
    }
};

void populateBoasToLLVMConversionPatterns(BoasToLLVMTypeConverter &typeConverter,
                                         ::mlir::RewritePatternSet &patterns) {
    patterns.add<ListCreateOpLowering,
                ListAppendOpLowering,
                ListNestedCreateOpLowering,
                ListGetOpLowering,
                NumberConstantOpLowering,
                PrintOpLowering,
                TensorCreateOpLowering,
                TensorMatMulOpLowering>(patterns.getContext());
}

struct ConvertBoasToLLVMPass
    : public ::mlir::PassWrapper<ConvertBoasToLLVMPass, ::mlir::OperationPass<::mlir::ModuleOp>> {
    void runOnOperation() override {
        auto module = getOperation();
        auto context = &getContext();
        
        BoasToLLVMTypeConverter typeConverter(context);
        ::mlir::RewritePatternSet patterns(context);
        ::mlir::ConversionTarget target(*context);
        
        target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
        target.addIllegalDialect<BoasDialect>();
        
        populateBoasToLLVMConversionPatterns(typeConverter, patterns);
        
        if (::mlir::failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createConvertBoasToLLVMPass() {
    return std::make_unique<ConvertBoasToLLVMPass>();
}

} // namespace mlir
} // namespace boas
