// MLIRGenTiming.cpp - 时间相关操作实现
#include "mlirops/MLIRGen.h"

namespace matrix {

mlir::Value MLIRGen::generateTimeNowMLIR(const TimeCallExprAST* expr) {
    auto loc = builder->getUnknownLoc();
    
    return builder->create<mlir::func::CallOp>(
        loc,
        "system_time_usec",
        mlir::TypeRange{builder->getF64Type()},
        mlir::ValueRange{}
    ).getResult(0);
}

mlir::Value MLIRGen::generateTimeDiffMLIR(mlir::Value lhs, mlir::Value rhs) {
    auto loc = builder->getUnknownLoc();
    auto diff = builder->create<mlir::arith::SubFOp>(loc, lhs, rhs);
    return diff;
}

} // namespace matrix