#ifndef BOAS_MLIR_GEN_H
#define BOAS_MLIR_GEN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "boas/ast/ast.h"

namespace boas {
// Forward declarations
class ListAST;
class NumberAST;
class PrintAST;
class BlockAST;

namespace mlir {

class MLIRGen {
public:
    explicit MLIRGen(::mlir::MLIRContext& context) 
        : builder(&context), module(::mlir::ModuleOp::create(builder.getUnknownLoc())) {}
    
    void generateModuleOp(AST* ast);
    ::mlir::ModuleOp getModule() { return module; }

private:
    ::mlir::Value mlirGen(const AST& ast);
    ::mlir::Value mlirGen(const ListAST& listAst);
    ::mlir::Value mlirGen(const NumberAST& numberAst);
    ::mlir::Value mlirGen(const PrintAST& printAst);

    ::mlir::OpBuilder builder;
    ::mlir::ModuleOp module;
};

} // namespace mlir
} // namespace boas

#endif // BOAS_MLIR_GEN_H
