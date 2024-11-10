#include "boas/backend/mlir/BoasToLLVM.h"
#include "boas/backend/mlir/ListOps.h"
#include "boas/backend/mlir/NumberOps.h"
#include "boas/backend/mlir/BoasDialect.h"
#include "boas/backend/mlir/PrintOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

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
            loc, elementTy, structTy, allocated, 
            ::mlir::ValueRange{zero, zero});
        rewriter.create<::mlir::LLVM::StoreOp>(loc, nullPtr, dataPtr);
        
        auto sizePtr = rewriter.create<::mlir::LLVM::GEPOp>(
            loc, rewriter.getI64Type(), structTy, allocated,
            ::mlir::ValueRange{zero, rewriter.create<::mlir::LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), 1)});
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
        
        // 获取列表和索引
        auto list = getOp.getList();
        auto index = getOp.getIndex();
        
        // 获取列表结构体类型
        auto elementTy = ::mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
        auto structTy = ::mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {elementTy, rewriter.getI64Type()}  // data pointer and size
        );
        
        // 创建GEP操作来访问数据指针
        auto zero = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), 0);
            
        // 访问结构体中的数据指针字段
        auto dataPtr = rewriter.create<::mlir::LLVM::GEPOp>(
            loc,
            elementTy,
            structTy,
            list,
            ::mlir::ValueRange{zero, zero},
            false);  // inbounds flag
            
        // 加载数据指针
        auto data = rewriter.create<::mlir::LLVM::LoadOp>(
            loc,
            elementTy,  // 结果类型
            dataPtr,    // 要加载的地址
            0);        // 对齐值
            
        // 使用索引访问元素
        auto element = rewriter.create<::mlir::LLVM::GEPOp>(
            loc,
            rewriter.getI64Type(),  // 结果类型
            rewriter.getI64Type(),  // 元素类型
            data,                   // 基址
            ::mlir::ValueRange{index},  // 索引
            false);                 // inbounds flag
            
        // 加载元素值
        auto result = rewriter.create<::mlir::LLVM::LoadOp>(
            loc,
            rewriter.getI64Type(),  // 结果类型
            element,                // 要加载的地址
            0);                     // 对齐值
            
        rewriter.replaceOp(op, result.getResult());
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
                PrintOpLowering>(patterns.getContext());
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
