module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @printFloat(f64) attributes {sym_visibility = "private"}
  llvm.func @printString(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @system_time_msec() -> f64 attributes {sym_visibility = "private"}
  llvm.func @generate_random() -> f64 attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %4 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %7 = llvm.mlir.constant(8.000000e+00 : f64) : f64
    %8 = llvm.mlir.constant(7.000000e+00 : f64) : f64
    %9 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %10 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(2 : index) : i64
    %13 = llvm.mlir.constant(2 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(4 : index) : i64
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.getelementptr %16[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.call @malloc(%18) : (i64) -> !llvm.ptr
    %20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.constant(0 : index) : i64
    %24 = llvm.insertvalue %23, %22[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %12, %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %13, %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %13, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %14, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.mlir.constant(2 : index) : i64
    %31 = llvm.mul %2, %30 : i64
    %32 = llvm.add %31, %2 : i64
    %33 = llvm.getelementptr %29[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %1, %33 : f64, !llvm.ptr
    %34 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(2 : index) : i64
    %36 = llvm.mul %2, %35 : i64
    %37 = llvm.add %36, %11 : i64
    %38 = llvm.getelementptr %34[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %3, %38 : f64, !llvm.ptr
    %39 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(2 : index) : i64
    %41 = llvm.mul %11, %40 : i64
    %42 = llvm.add %41, %2 : i64
    %43 = llvm.getelementptr %39[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %3, %43 : f64, !llvm.ptr
    %44 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.mlir.constant(2 : index) : i64
    %46 = llvm.mul %11, %45 : i64
    %47 = llvm.add %46, %11 : i64
    %48 = llvm.getelementptr %44[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %10, %48 : f64, !llvm.ptr
    %49 = llvm.mlir.constant(2 : index) : i64
    %50 = llvm.mlir.constant(2 : index) : i64
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.mlir.constant(4 : index) : i64
    %53 = llvm.mlir.zero : !llvm.ptr
    %54 = llvm.getelementptr %53[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.call @malloc(%55) : (i64) -> !llvm.ptr
    %57 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.insertvalue %56, %57[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %49, %61[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.insertvalue %50, %62[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %50, %63[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %51, %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.extractvalue %65[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mlir.constant(2 : index) : i64
    %68 = llvm.mul %2, %67 : i64
    %69 = llvm.add %68, %2 : i64
    %70 = llvm.getelementptr %66[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %9, %70 : f64, !llvm.ptr
    %71 = llvm.extractvalue %65[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(2 : index) : i64
    %73 = llvm.mul %2, %72 : i64
    %74 = llvm.add %73, %11 : i64
    %75 = llvm.getelementptr %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %1, %75 : f64, !llvm.ptr
    %76 = llvm.extractvalue %65[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.mlir.constant(2 : index) : i64
    %78 = llvm.mul %11, %77 : i64
    %79 = llvm.add %78, %2 : i64
    %80 = llvm.getelementptr %76[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %8, %80 : f64, !llvm.ptr
    %81 = llvm.extractvalue %65[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(2 : index) : i64
    %83 = llvm.mul %11, %82 : i64
    %84 = llvm.add %83, %11 : i64
    %85 = llvm.getelementptr %81[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %7, %85 : f64, !llvm.ptr
    %86 = llvm.mlir.constant(2 : index) : i64
    %87 = llvm.mlir.constant(2 : index) : i64
    %88 = llvm.mlir.constant(1 : index) : i64
    %89 = llvm.mlir.constant(4 : index) : i64
    %90 = llvm.mlir.zero : !llvm.ptr
    %91 = llvm.getelementptr %90[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.call @malloc(%92) : (i64) -> !llvm.ptr
    %94 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %95 = llvm.insertvalue %93, %94[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.insertvalue %93, %95[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.mlir.constant(0 : index) : i64
    %98 = llvm.insertvalue %97, %96[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.insertvalue %86, %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.insertvalue %87, %99[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.insertvalue %87, %100[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.insertvalue %88, %101[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%2 : i64)
  ^bb1(%103: i64):  // 2 preds: ^bb0, ^bb4
    %104 = llvm.icmp "slt" %103, %5 : i64
    llvm.cond_br %104, ^bb2(%2 : i64), ^bb5(%2 : i64)
  ^bb2(%105: i64):  // 2 preds: ^bb1, ^bb3
    %106 = llvm.icmp "slt" %105, %5 : i64
    llvm.cond_br %106, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %107 = llvm.extractvalue %102[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.constant(2 : index) : i64
    %109 = llvm.mul %103, %108 : i64
    %110 = llvm.add %109, %105 : i64
    %111 = llvm.getelementptr %107[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %6, %111 : f64, !llvm.ptr
    %112 = llvm.add %105, %11 : i64
    llvm.br ^bb2(%112 : i64)
  ^bb4:  // pred: ^bb2
    %113 = llvm.add %103, %11 : i64
    llvm.br ^bb1(%113 : i64)
  ^bb5(%114: i64):  // 2 preds: ^bb1, ^bb10
    %115 = llvm.icmp "slt" %114, %5 : i64
    llvm.cond_br %115, ^bb6(%2 : i64), ^bb11
  ^bb6(%116: i64):  // 2 preds: ^bb5, ^bb9
    %117 = llvm.icmp "slt" %116, %5 : i64
    llvm.cond_br %117, ^bb7(%2 : i64), ^bb10
  ^bb7(%118: i64):  // 2 preds: ^bb6, ^bb8
    %119 = llvm.icmp "slt" %118, %5 : i64
    llvm.cond_br %119, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %120 = llvm.extractvalue %102[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.mlir.constant(2 : index) : i64
    %122 = llvm.mul %114, %121 : i64
    %123 = llvm.add %122, %116 : i64
    %124 = llvm.getelementptr %120[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %125 = llvm.load %124 : !llvm.ptr -> f64
    %126 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.constant(2 : index) : i64
    %128 = llvm.mul %114, %127 : i64
    %129 = llvm.add %128, %118 : i64
    %130 = llvm.getelementptr %126[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %131 = llvm.load %130 : !llvm.ptr -> f64
    %132 = llvm.extractvalue %65[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.mlir.constant(2 : index) : i64
    %134 = llvm.mul %118, %133 : i64
    %135 = llvm.add %134, %116 : i64
    %136 = llvm.getelementptr %132[%135] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %137 = llvm.load %136 : !llvm.ptr -> f64
    %138 = llvm.fmul %131, %137  : f64
    %139 = llvm.fadd %125, %138  : f64
    %140 = llvm.extractvalue %102[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.mlir.constant(2 : index) : i64
    %142 = llvm.mul %114, %141 : i64
    %143 = llvm.add %142, %116 : i64
    %144 = llvm.getelementptr %140[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %139, %144 : f64, !llvm.ptr
    %145 = llvm.add %118, %11 : i64
    llvm.br ^bb7(%145 : i64)
  ^bb9:  // pred: ^bb7
    %146 = llvm.add %116, %11 : i64
    llvm.br ^bb6(%146 : i64)
  ^bb10:  // pred: ^bb6
    %147 = llvm.add %114, %11 : i64
    llvm.br ^bb5(%147 : i64)
  ^bb11:  // pred: ^bb5
    llvm.call @printFloat(%4) : (f64) -> ()
    llvm.br ^bb12(%2 : i64)
  ^bb12(%148: i64):  // 2 preds: ^bb11, ^bb15
    %149 = llvm.icmp "slt" %148, %5 : i64
    llvm.cond_br %149, ^bb13(%2 : i64), ^bb16
  ^bb13(%150: i64):  // 2 preds: ^bb12, ^bb14
    %151 = llvm.icmp "slt" %150, %5 : i64
    llvm.cond_br %151, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %152 = llvm.extractvalue %102[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.mlir.constant(2 : index) : i64
    %154 = llvm.mul %148, %153 : i64
    %155 = llvm.add %154, %150 : i64
    %156 = llvm.getelementptr %152[%155] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %157 = llvm.load %156 : !llvm.ptr -> f64
    llvm.call @printFloat(%157) : (f64) -> ()
    %158 = llvm.add %150, %11 : i64
    llvm.br ^bb13(%158 : i64)
  ^bb15:  // pred: ^bb13
    %159 = llvm.add %148, %11 : i64
    llvm.br ^bb12(%159 : i64)
  ^bb16:  // pred: ^bb12
    llvm.return %0 : i32
  }
}

