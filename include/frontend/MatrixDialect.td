#ifndef MATRIX_DIALECT
#define MATRIX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Matrix_Dialect : Dialect {
  let name = "matrix";
  let summary = "A dialect for Matrix language";
  let description = [{
    The Matrix dialect is designed for matrix operations and tensor computations.
  }];
  let cppNamespace = "::matrix";
}

// 基本类型定义
def Matrix_Type : TypeDef<Matrix_Dialect, "Matrix"> {
  let mnemonic = "matrix";
}

def Tensor_Type : TypeDef<Matrix_Dialect, "Tensor"> {
  let mnemonic = "tensor";
}

// 操作定义
class Matrix_Op<string mnemonic, list<Trait> traits = []> :
    Op<Matrix_Dialect, mnemonic, traits>;

def ImportOp : Matrix_Op<"import"> {
  let summary = "Import operation";
  let description = [{
    The import operation represents module imports.
    
    Example:
      %0 = matrix.import "tensor"
  }];
  
  let arguments = (ins StrAttr:$moduleName);
  let results = (outs);
  
  let assemblyFormat = [{
    $moduleName attr-dict
  }];
}

def FromImportOp : Matrix_Op<"from_import"> {
  let summary = "From import operation";
  let description = [{
    The from import operation represents selective imports.
    
    Example:
      %0 = matrix.from_import "tensor" "matmul"
  }];
  
  let arguments = (ins StrAttr:$moduleName, StrAttr:$funcName);
  let results = (outs);
  
  let assemblyFormat = [{
    $moduleName $funcName attr-dict
  }];
}

def TensorCreateOp : Matrix_Op<"tensor_create"> {
  let summary = "Tensor creation operation";
  let description = [{
    Creates a new tensor with given data.
    
    Example:
      %0 = matrix.tensor_create [[1, 2], [3, 4]]
  }];
  
  let arguments = (ins AnyAttr:$data);
  let results = (outs Tensor_Type:$result);
  
  let assemblyFormat = [{
    $data attr-dict
  }];
}

def MatMulOp : Matrix_Op<"matmul"> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication between two tensors.
    
    Example:
      %2 = matrix.matmul %0, %1
  }];
  
  let arguments = (ins Tensor_Type:$lhs, Tensor_Type:$rhs);
  let results = (outs Tensor_Type:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
  }];
}

def PrintOp : Matrix_Op<"print"> {
  let summary = "Print operation";
  let description = [{
    Prints a value.
    
    Example:
      matrix.print %0
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs);
  
  let assemblyFormat = [{
    $value attr-dict
  }];
}

#endif // MATRIX_DIALECT 