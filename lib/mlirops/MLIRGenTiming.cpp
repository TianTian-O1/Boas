// MLIRGenTiming.cpp - 时间相关操作实现
#include "mlirops/MLIRGen.h"

namespace matrix {

mlir::Value MLIRGen::generateTimeNowMLIR(const TimeCallExprAST* expr) {
    auto loc = builder->getUnknownLoc();
    
    return builder->create<mlir::func::CallOp>(
        loc,
        "system_time_msec",
        mlir::TypeRange{builder->getF64Type()},
        mlir::ValueRange{}
    ).getResult(0);
}

mlir::Value MLIRGen::generateTimeDiffMLIR(mlir::Value lhs, mlir::Value rhs) {
    auto loc = builder->getUnknownLoc();
    return builder->create<mlir::arith::SubFOp>(loc, lhs, rhs);
}

mlir::Value MLIRGen::convertToMilliseconds(mlir::Value seconds) {
    auto loc = builder->getUnknownLoc();
    auto scale = createConstantF64(1000.0);
    return builder->create<mlir::arith::MulFOp>(loc, seconds, scale);
}

} // namespace matrix