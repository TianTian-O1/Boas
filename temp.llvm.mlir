module attributes {sym_visibility = "private"} {
  llvm.mlir.global private @module_name_main(dense<[109, 97, 105, 110, 0]> : tensor<5xi8>) {addr_space = 0 : i32} : !llvm.array<5 x i8>
  llvm.mlir.global private @module_name_benchmark(dense<[98, 101, 110, 99, 104, 109, 97, 114, 107, 0]> : tensor<10xi8>) {addr_space = 0 : i32} : !llvm.array<10 x i8>
  llvm.mlir.global private @module_name_tensor(dense<[116, 101, 110, 115, 111, 114, 0]> : tensor<7xi8>) {addr_space = 0 : i32} : !llvm.array<7 x i8>
  llvm.func @import_module(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64) attributes {sym_visibility = "private"} {
    llvm.return
  }
  %0 = llvm.mlir.constant(7 : index) : i64
  %1 = llvm.mlir.constant(1 : index) : i64
  %2 = llvm.mlir.zero : !llvm.ptr
  %3 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
  %5 = llvm.mlir.addressof @module_name_tensor : !llvm.ptr
  %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
  %7 = llvm.mlir.constant(3735928559 : index) : i64
  %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
  %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %11 = llvm.insertvalue %6, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %12 = llvm.mlir.constant(0 : index) : i64
  %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %14 = llvm.insertvalue %0, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %15 = llvm.insertvalue %1, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %16 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %17 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %18 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %19 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %20 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  llvm.call @import_module(%16, %17, %18, %19, %20) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
  %21 = llvm.mlir.constant(10 : index) : i64
  %22 = llvm.mlir.constant(1 : index) : i64
  %23 = llvm.mlir.zero : !llvm.ptr
  %24 = llvm.getelementptr %23[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
  %26 = llvm.mlir.addressof @module_name_benchmark : !llvm.ptr
  %27 = llvm.getelementptr %26[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
  %28 = llvm.mlir.constant(3735928559 : index) : i64
  %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
  %30 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %32 = llvm.insertvalue %27, %31[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %33 = llvm.mlir.constant(0 : index) : i64
  %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %35 = llvm.insertvalue %21, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %36 = llvm.insertvalue %22, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %37 = llvm.extractvalue %36[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %38 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %39 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %40 = llvm.extractvalue %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %41 = llvm.extractvalue %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  llvm.call @import_module(%37, %38, %39, %40, %41) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
  %42 = llvm.mlir.constant(5 : index) : i64
  %43 = llvm.mlir.constant(1 : index) : i64
  %44 = llvm.mlir.zero : !llvm.ptr
  %45 = llvm.getelementptr %44[%42] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
  %47 = llvm.mlir.addressof @module_name_main : !llvm.ptr
  %48 = llvm.getelementptr %47[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
  %49 = llvm.mlir.constant(3735928559 : index) : i64
  %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
  %51 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %52 = llvm.insertvalue %50, %51[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %53 = llvm.insertvalue %48, %52[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %54 = llvm.mlir.constant(0 : index) : i64
  %55 = llvm.insertvalue %54, %53[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %56 = llvm.insertvalue %42, %55[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %57 = llvm.insertvalue %43, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %58 = llvm.extractvalue %57[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %59 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %60 = llvm.extractvalue %57[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %61 = llvm.extractvalue %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %62 = llvm.extractvalue %57[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  llvm.call @import_module(%58, %59, %60, %61, %62) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
}

