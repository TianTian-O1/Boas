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

struct TensorCreateOpLowering : public ::mlir::ConversionPattern {
    explicit TensorCreateOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(TensorCreateOp::getOperationName(), 1, ctx) {}
        
    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto createOp = cast<TensorCreateOp>(op);
        auto loc = op->getLoc();
        
        // 获取tensor的shape
        auto shape = createOp->getAttrOfType<::mlir::ArrayAttr>("shape");
        int64_t size = 1;
        for (auto dim : shape) {
            size *= dim.cast<::mlir::IntegerAttr>().getInt();
        }
        
        // 分配内存
        auto mallocSize = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), size * sizeof(double));
            
        auto mallocRef = ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "malloc");
        auto allocated = rewriter.create<::mlir::LLVM::CallOp>(
            loc,
            ::mlir::TypeRange{::mlir::LLVM::LLVMPointerType::get(rewriter.getContext())},
            mallocRef,
            ::mlir::ValueRange{mallocSize}).getResult();
            
        rewriter.replaceOp(op, allocated);
        return ::mlir::success();
    }
};

struct TensorMatMulOpLowering : public ::mlir::ConversionPattern {
    explicit TensorMatMulOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(TensorMatMulOp::getOperationName(), 1, ctx) {}
        
    ::mlir::LogicalResult
    matchAndRewrite(::mlir::Operation *op, ArrayRef<::mlir::Value> operands,
                   ::mlir::ConversionPatternRewriter &rewriter) const override {
        auto matmulOp = cast<TensorMatMulOp>(op);
        auto loc = op->getLoc();
        
        // 获取输入矩阵的维度
        auto lhsType = matmulOp.getOperand(0).getType().cast<TensorType>();
        auto rhsType = matmulOp.getOperand(1).getType().cast<TensorType>();
        
        auto m = lhsType.getShape()[0];
        auto k = lhsType.getShape()[1];
        auto n = rhsType.getShape()[1];
        
        // 分配结果矩阵内存
        auto resultSize = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), m * n * sizeof(double));
            
        auto mallocRef = ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "malloc");
        auto result = rewriter.create<::mlir::LLVM::CallOp>(
            loc,
            ::mlir::TypeRange{::mlir::LLVM::LLVMPointerType::get(rewriter.getContext())},
            mallocRef,
            ::mlir::ValueRange{resultSize}).getResult();
            
        // 创建基本块和循环变量
        auto *currentBlock = rewriter.getBlock();
        auto *iLoopHeader = rewriter.createBlock(currentBlock->getParent());
        auto *iLoopBody = rewriter.createBlock(currentBlock->getParent());
        auto *jLoopHeader = rewriter.createBlock(currentBlock->getParent());
        auto *jLoopBody = rewriter.createBlock(currentBlock->getParent());
        auto *kLoopHeader = rewriter.createBlock(currentBlock->getParent());
        auto *kLoopBody = rewriter.createBlock(currentBlock->getParent());
        auto *exitBlock = rewriter.createBlock(currentBlock->getParent());
        
        // i循环初始化
        auto zero = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), 0);
        auto one = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), 1);
        auto mValue = rewriter.create<::mlir::LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), m);
            
        rewriter.create<::mlir::LLVM::BrOp>(loc, iLoopHeader);
        
        // i循环头
        rewriter.setInsertionPointToEnd(iLoopHeader);
        auto iVar = rewriter.create<::mlir::LLVM::PHIOp>(
            loc, rewriter.getI64Type(), 
            ::mlir::ValueRange{zero, rewriter.create<::mlir::LLVM::AddOp>(
                loc, rewriter.getI64Type(), iVar, one)});
        auto iCond = rewriter.create<::mlir::LLVM::ICmpOp>(
            loc, ::mlir::LLVM::ICmpPredicate::slt, iVar, mValue);
        rewriter.create<::mlir::LLVM::CondBrOp>(
            loc, iCond, iLoopBody, exitBlock);
            
        // i循环体 (类似地实现j和k循环)
        rewriter.setInsertionPointToEnd(iLoopBody);
        // ... 实现j循环和k循环的逻辑
        
        // 矩阵乘法核心计算
        auto aIndex = rewriter.create<::mlir::LLVM::MulOp>(
            loc, iVar, rewriter.create<::mlir::LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), k));
        
        // 计算B[p,j]的地址
        auto bIndex = rewriter.create<::mlir::LLVM::MulOp>(
            loc, kVar, rewriter.create<::mlir::LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), n));
                
        auto aElem = rewriter.create<::mlir::LLVM::LoadOp>(
            loc, rewriter.getF64Type(), 
            matmulOp.getOperand(0), aIndex);
        auto bElem = rewriter.create<::mlir::LLVM::LoadOp>(
            loc, rewriter.getF64Type(),
            matmulOp.getOperand(1), bIndex);
                
        // 计算乘积并累加
        auto prod = rewriter.create<::mlir::LLVM::FMulOp>(
            loc, aElem, bElem);
        sum = rewriter.create<::mlir::LLVM::FAddOp>(loc, sum, prod);
        
        // 计算结果矩阵C[i,j]的地址
        auto cIndex = rewriter.create<::mlir::LLVM::MulOp>(
            loc, iVar, rewriter.create<::mlir::LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), n));
        cIndex = rewriter.create<::mlir::LLVM::AddOp>(loc, cIndex, jValue);
                
        // 存储结果
        rewriter.create<::mlir::LLVM::StoreOp>(loc, sum, result, cIndex);
        
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
