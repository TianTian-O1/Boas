#include "boas/backend/mlir/MLIRGen.h"
#include "boas/ast/list_ast.h"
#include "boas/ast/number_ast.h"
#include "boas/ast/print_ast.h"
#include "boas/backend/mlir/ListOps.h"
#include "boas/backend/mlir/NumberOps.h"
#include "boas/backend/mlir/PrintOps.h"
#include "boas/ast/block_ast.h"

namespace boas {
namespace mlir {

void MLIRGen::generateModuleOp(AST* ast) {
    builder.setInsertionPointToStart(module.getBody());
    if (auto* blockAst = llvm::dyn_cast<BlockAST>(ast)) {
        for (const auto& stmt : blockAst->getStatements()) {
            mlirGen(*stmt);
        }
    } else {
        mlirGen(*ast);
    }
}

::mlir::Value MLIRGen::mlirGen(const AST& ast) {
    if (auto* listAst = llvm::dyn_cast<ListAST>(&ast)) {
        return mlirGen(*listAst);
    } else if (auto* numberAst = llvm::dyn_cast<NumberAST>(&ast)) {
        return mlirGen(*numberAst);
    } else if (auto* printAst = llvm::dyn_cast<PrintAST>(&ast)) {
        return mlirGen(*printAst);
    }
    llvm_unreachable("Unhandled AST node");
}

::mlir::Value MLIRGen::mlirGen(const ListAST& listAst) {
    if (listAst.isNested()) {
        auto listOp = builder.create<ListNestedCreateOp>(builder.getUnknownLoc());
        
        for (const auto& element : listAst.getElements()) {
            if (auto* nestedList = llvm::dyn_cast<ListAST>(element.get())) {
                auto nestedValue = mlirGen(*nestedList);
                builder.create<ListAppendOp>(
                    builder.getUnknownLoc(),
                    listOp.getResult(),
                    nestedValue
                );
            }
        }
        return listOp.getResult();
    } else if (listAst.hasIndex()) {
        auto list = mlirGen(*listAst.getSourceList());
        
        auto index = builder.create<NumberConstantOp>(
            builder.getUnknownLoc(),
            listAst.getIndex()
        );
        
        return builder.create<ListGetOp>(
            builder.getUnknownLoc(),
            list,
            index.getResult()
        ).getResult();
    }
    
    auto listOp = builder.create<ListCreateOp>(builder.getUnknownLoc());
    
    for (const auto& element : listAst.getElements()) {
        if (auto* nestedList = llvm::dyn_cast<ListAST>(element.get())) {
            // Handle nested list
            auto nestedValue = mlirGen(*nestedList);
            builder.create<ListAppendOp>(
                builder.getUnknownLoc(),
                listOp.getResult(),
                nestedValue
            );
        } else if (auto* numberAst = llvm::dyn_cast<NumberAST>(element.get())) {
            // Handle number element
            auto value = mlirGen(*numberAst);
            builder.create<ListAppendOp>(
                builder.getUnknownLoc(),
                listOp.getResult(),
                value
            );
        }
    }
    
    return listOp.getResult();
}

::mlir::Value MLIRGen::mlirGen(const NumberAST& numberAst) {
    auto op = builder.create<NumberConstantOp>(
        builder.getUnknownLoc(),
        numberAst.getValue()
    );
    return op.getResult();
}

::mlir::Value MLIRGen::mlirGen(const PrintAST& printAst) {
    auto value = mlirGen(*printAst.getValue());
    auto printOp = builder.create<PrintOp>(
        builder.getUnknownLoc(),
        value
    );
    return printOp.getResult();
}

} // namespace mlir
} // namespace boas
