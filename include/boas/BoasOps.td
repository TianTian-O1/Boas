#ifndef BOAS_OPS
#define BOAS_OPS

include "mlir/IR/OpBase.td"

def Boas_Dialect : Dialect {
  let name = "boas";
  let summary = "A dialect for Boas language";
  let cppNamespace = "::boas";
}

def PrintOp : Op<Boas_Dialect, "print"> {
  let summary = "Print operation";
  let arguments = (ins StrAttr:$value);
  let assemblyFormat = "$value attr-dict";
}

#endif // BOAS_OPS
