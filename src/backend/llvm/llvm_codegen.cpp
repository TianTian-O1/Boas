#include "boas/backend/mlir/Conversion/BoasToLLVM.h"
#include "boas/backend/mlir/ListOps.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace boas {
namespace mlir {

struct ListCreateOpLowering : public ::mlir::ConversionPattern {
    explicit ListCreateOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ListCreateOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        // 将ListCreateOp转换为LLVM IR的malloc调用
        auto listOp = cast<ListCreateOp>(op);
        auto loc = op->getLoc();
        
        // 创建LLVM类型
        auto structTy = LLVM::LLVMStructType::getIdentified(op->getContext(), "list");
        auto mallocSize = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), sizeof(void*));
            
        // 调用malloc
        auto mallocFunc = LLVM::lookupOrCreateMallocFn(op->getParentOfType<ModuleOp>());
        auto allocated = rewriter.create<LLVM::CallOp>(
            loc, mallocFunc, mallocSize).getResult(0);
            
        // 初始化list结构
        rewriter.replaceOp(op, allocated);
        return success();
    }
};

struct ListAppendOpLowering : public ::mlir::ConversionPattern {
    explicit ListAppendOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ListAppendOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto appendOp = cast<ListAppendOp>(op);
        auto loc = op->getLoc();
        
        // 实现append逻辑
        // 1. 重新分配内存
        // 2. 复制现有元素
        // 3. 添加新元素
        
        return success();
    }
};

struct ListGetOpLowering : public ::mlir::ConversionPattern {
    explicit ListGetOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ListGetOp::getOperationName(), 1, ctx) {}

    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto getOp = cast<ListGetOp>(op);
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
            elementTy,
            dataPtr,
            0);
            
        // 使用索引访问元素
        auto element = rewriter.create<::mlir::LLVM::GEPOp>(
            loc,
            rewriter.getI64Type(),
            rewriter.getI64Type(),
            data,
            ::mlir::ValueRange{index},
            false);
            
        // 加载元素值
        auto result = rewriter.create<::mlir::LLVM::LoadOp>(
            loc,
            rewriter.getI64Type(),
            element,
            0);
            
        rewriter.replaceOp(op, result.getResult());
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
        
        // 获取要打印的值
        auto value = printOp.getOperand();
        
        // 创建 printf 函数声明
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        auto printfFunc = LLVM::lookupOrCreatePrintfFn(moduleOp);
        
        // 创建格式字符串
        auto formatStr = rewriter.create<LLVM::GlobalOp>(
            loc, 
            LLVM::LLVMArrayType::get(rewriter.getI8Type(), 4),
            /*isConstant=*/true,
            LLVM::Linkage::Internal,
            "frmt",
            rewriter.getStringAttr("%d\n"));
            
        // 调用 printf
        auto formatPtr = rewriter.create<LLVM::GEPOp>(
            loc,
            LLVM::LLVMPointerType::get(rewriter.getContext()),
            formatStr,
            ArrayRef<Value>{
                rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0),
                rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0)
            });
            
        rewriter.create<LLVM::CallOp>(
            loc,
            printfFunc,
            ArrayRef<Value>{formatPtr, value});
            
        rewriter.eraseOp(op);
        return ::mlir::success();
    }
};

void populateBoasToLLVMConversionPatterns(BoasToLLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
    patterns.add<ListCreateOpLowering, ListAppendOpLowering, 
                ListGetOpLowering, PrintOpLowering>(patterns.getContext());
}

struct ConvertBoasToLLVMPass
    : public ::mlir::PassWrapper<ConvertBoasToLLVMPass, ::mlir::OperationPass<::mlir::ModuleOp>> {
    void runOnOperation() override {
        auto module = getOperation();
        auto context = &getContext();
        
        BoasToLLVMTypeConverter typeConverter(context);
        RewritePatternSet patterns(context);
        ConversionTarget target(*context);
        
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addIllegalDialect<BoasDialect>();
        
        populateBoasToLLVMConversionPatterns(typeConverter, patterns);
        
        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<::mlir::Pass> createConvertBoasToLLVMPass() {
    return std::make_unique<ConvertBoasToLLVMPass>();
}

} // namespace mlir
} // namespace boas
