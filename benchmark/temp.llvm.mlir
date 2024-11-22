module {
  llvm.func @printFloat(f64) attributes {sym_visibility = "private"}
  llvm.func @printString(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @system_time_msec() -> f64 attributes {sym_visibility = "private"}
  llvm.func @generate_random() -> f64 attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %3 = llvm.mlir.constant(1024 : index) : i64
    %4 = llvm.mlir.constant(9.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(8.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(512 : index) : i64
    %7 = llvm.mlir.constant(7.000000e+00 : f64) : f64
    %8 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    %9 = llvm.mlir.constant(256 : index) : i64
    %10 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %11 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %12 = llvm.mlir.constant(128 : index) : i64
    %13 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %14 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(64 : index) : i64
    %17 = llvm.mlir.constant(0 : index) : i64
    llvm.call @printFloat(%0) : (f64) -> ()
    llvm.br ^bb1(%17 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb4
    %19 = llvm.icmp "slt" %18, %16 : i64
    llvm.cond_br %19, ^bb2(%17 : i64), ^bb5(%17 : i64)
  ^bb2(%20: i64):  // 2 preds: ^bb1, ^bb3
    %21 = llvm.icmp "slt" %20, %16 : i64
    llvm.cond_br %21, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %22 = llvm.call @generate_random() : () -> f64
    %23 = llvm.add %20, %15 : i64
    llvm.br ^bb2(%23 : i64)
  ^bb4:  // pred: ^bb2
    %24 = llvm.add %18, %15 : i64
    llvm.br ^bb1(%24 : i64)
  ^bb5(%25: i64):  // 2 preds: ^bb1, ^bb8
    %26 = llvm.icmp "slt" %25, %16 : i64
    llvm.cond_br %26, ^bb6(%17 : i64), ^bb9(%17 : i64)
  ^bb6(%27: i64):  // 2 preds: ^bb5, ^bb7
    %28 = llvm.icmp "slt" %27, %16 : i64
    llvm.cond_br %28, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %29 = llvm.call @generate_random() : () -> f64
    %30 = llvm.add %27, %15 : i64
    llvm.br ^bb6(%30 : i64)
  ^bb8:  // pred: ^bb6
    %31 = llvm.add %25, %15 : i64
    llvm.br ^bb5(%31 : i64)
  ^bb9(%32: i64):  // 2 preds: ^bb5, ^bb14
    %33 = llvm.icmp "slt" %32, %16 : i64
    llvm.cond_br %33, ^bb10(%17 : i64), ^bb15
  ^bb10(%34: i64):  // 2 preds: ^bb9, ^bb13
    %35 = llvm.icmp "slt" %34, %16 : i64
    llvm.cond_br %35, ^bb11(%17 : i64), ^bb14
  ^bb11(%36: i64):  // 2 preds: ^bb10, ^bb12
    %37 = llvm.icmp "slt" %36, %16 : i64
    llvm.cond_br %37, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %38 = llvm.add %36, %15 : i64
    llvm.br ^bb11(%38 : i64)
  ^bb13:  // pred: ^bb11
    %39 = llvm.add %34, %15 : i64
    llvm.br ^bb10(%39 : i64)
  ^bb14:  // pred: ^bb10
    %40 = llvm.add %32, %15 : i64
    llvm.br ^bb9(%40 : i64)
  ^bb15:  // pred: ^bb9
    llvm.call @printFloat(%14) : (f64) -> ()
    llvm.call @printFloat(%13) : (f64) -> ()
    llvm.br ^bb16(%17 : i64)
  ^bb16(%41: i64):  // 2 preds: ^bb15, ^bb19
    %42 = llvm.icmp "slt" %41, %12 : i64
    llvm.cond_br %42, ^bb17(%17 : i64), ^bb20(%17 : i64)
  ^bb17(%43: i64):  // 2 preds: ^bb16, ^bb18
    %44 = llvm.icmp "slt" %43, %12 : i64
    llvm.cond_br %44, ^bb18, ^bb19
  ^bb18:  // pred: ^bb17
    %45 = llvm.call @generate_random() : () -> f64
    %46 = llvm.add %43, %15 : i64
    llvm.br ^bb17(%46 : i64)
  ^bb19:  // pred: ^bb17
    %47 = llvm.add %41, %15 : i64
    llvm.br ^bb16(%47 : i64)
  ^bb20(%48: i64):  // 2 preds: ^bb16, ^bb23
    %49 = llvm.icmp "slt" %48, %12 : i64
    llvm.cond_br %49, ^bb21(%17 : i64), ^bb24(%17 : i64)
  ^bb21(%50: i64):  // 2 preds: ^bb20, ^bb22
    %51 = llvm.icmp "slt" %50, %12 : i64
    llvm.cond_br %51, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %52 = llvm.call @generate_random() : () -> f64
    %53 = llvm.add %50, %15 : i64
    llvm.br ^bb21(%53 : i64)
  ^bb23:  // pred: ^bb21
    %54 = llvm.add %48, %15 : i64
    llvm.br ^bb20(%54 : i64)
  ^bb24(%55: i64):  // 2 preds: ^bb20, ^bb29
    %56 = llvm.icmp "slt" %55, %12 : i64
    llvm.cond_br %56, ^bb25(%17 : i64), ^bb30
  ^bb25(%57: i64):  // 2 preds: ^bb24, ^bb28
    %58 = llvm.icmp "slt" %57, %12 : i64
    llvm.cond_br %58, ^bb26(%17 : i64), ^bb29
  ^bb26(%59: i64):  // 2 preds: ^bb25, ^bb27
    %60 = llvm.icmp "slt" %59, %12 : i64
    llvm.cond_br %60, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %61 = llvm.add %59, %15 : i64
    llvm.br ^bb26(%61 : i64)
  ^bb28:  // pred: ^bb26
    %62 = llvm.add %57, %15 : i64
    llvm.br ^bb25(%62 : i64)
  ^bb29:  // pred: ^bb25
    %63 = llvm.add %55, %15 : i64
    llvm.br ^bb24(%63 : i64)
  ^bb30:  // pred: ^bb24
    llvm.call @printFloat(%11) : (f64) -> ()
    llvm.call @printFloat(%10) : (f64) -> ()
    llvm.br ^bb31(%17 : i64)
  ^bb31(%64: i64):  // 2 preds: ^bb30, ^bb34
    %65 = llvm.icmp "slt" %64, %9 : i64
    llvm.cond_br %65, ^bb32(%17 : i64), ^bb35(%17 : i64)
  ^bb32(%66: i64):  // 2 preds: ^bb31, ^bb33
    %67 = llvm.icmp "slt" %66, %9 : i64
    llvm.cond_br %67, ^bb33, ^bb34
  ^bb33:  // pred: ^bb32
    %68 = llvm.call @generate_random() : () -> f64
    %69 = llvm.add %66, %15 : i64
    llvm.br ^bb32(%69 : i64)
  ^bb34:  // pred: ^bb32
    %70 = llvm.add %64, %15 : i64
    llvm.br ^bb31(%70 : i64)
  ^bb35(%71: i64):  // 2 preds: ^bb31, ^bb38
    %72 = llvm.icmp "slt" %71, %9 : i64
    llvm.cond_br %72, ^bb36(%17 : i64), ^bb39(%17 : i64)
  ^bb36(%73: i64):  // 2 preds: ^bb35, ^bb37
    %74 = llvm.icmp "slt" %73, %9 : i64
    llvm.cond_br %74, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    %75 = llvm.call @generate_random() : () -> f64
    %76 = llvm.add %73, %15 : i64
    llvm.br ^bb36(%76 : i64)
  ^bb38:  // pred: ^bb36
    %77 = llvm.add %71, %15 : i64
    llvm.br ^bb35(%77 : i64)
  ^bb39(%78: i64):  // 2 preds: ^bb35, ^bb44
    %79 = llvm.icmp "slt" %78, %9 : i64
    llvm.cond_br %79, ^bb40(%17 : i64), ^bb45
  ^bb40(%80: i64):  // 2 preds: ^bb39, ^bb43
    %81 = llvm.icmp "slt" %80, %9 : i64
    llvm.cond_br %81, ^bb41(%17 : i64), ^bb44
  ^bb41(%82: i64):  // 2 preds: ^bb40, ^bb42
    %83 = llvm.icmp "slt" %82, %9 : i64
    llvm.cond_br %83, ^bb42, ^bb43
  ^bb42:  // pred: ^bb41
    %84 = llvm.add %82, %15 : i64
    llvm.br ^bb41(%84 : i64)
  ^bb43:  // pred: ^bb41
    %85 = llvm.add %80, %15 : i64
    llvm.br ^bb40(%85 : i64)
  ^bb44:  // pred: ^bb40
    %86 = llvm.add %78, %15 : i64
    llvm.br ^bb39(%86 : i64)
  ^bb45:  // pred: ^bb39
    llvm.call @printFloat(%8) : (f64) -> ()
    llvm.call @printFloat(%7) : (f64) -> ()
    llvm.br ^bb46(%17 : i64)
  ^bb46(%87: i64):  // 2 preds: ^bb45, ^bb49
    %88 = llvm.icmp "slt" %87, %6 : i64
    llvm.cond_br %88, ^bb47(%17 : i64), ^bb50(%17 : i64)
  ^bb47(%89: i64):  // 2 preds: ^bb46, ^bb48
    %90 = llvm.icmp "slt" %89, %6 : i64
    llvm.cond_br %90, ^bb48, ^bb49
  ^bb48:  // pred: ^bb47
    %91 = llvm.call @generate_random() : () -> f64
    %92 = llvm.add %89, %15 : i64
    llvm.br ^bb47(%92 : i64)
  ^bb49:  // pred: ^bb47
    %93 = llvm.add %87, %15 : i64
    llvm.br ^bb46(%93 : i64)
  ^bb50(%94: i64):  // 2 preds: ^bb46, ^bb53
    %95 = llvm.icmp "slt" %94, %6 : i64
    llvm.cond_br %95, ^bb51(%17 : i64), ^bb54(%17 : i64)
  ^bb51(%96: i64):  // 2 preds: ^bb50, ^bb52
    %97 = llvm.icmp "slt" %96, %6 : i64
    llvm.cond_br %97, ^bb52, ^bb53
  ^bb52:  // pred: ^bb51
    %98 = llvm.call @generate_random() : () -> f64
    %99 = llvm.add %96, %15 : i64
    llvm.br ^bb51(%99 : i64)
  ^bb53:  // pred: ^bb51
    %100 = llvm.add %94, %15 : i64
    llvm.br ^bb50(%100 : i64)
  ^bb54(%101: i64):  // 2 preds: ^bb50, ^bb59
    %102 = llvm.icmp "slt" %101, %6 : i64
    llvm.cond_br %102, ^bb55(%17 : i64), ^bb60
  ^bb55(%103: i64):  // 2 preds: ^bb54, ^bb58
    %104 = llvm.icmp "slt" %103, %6 : i64
    llvm.cond_br %104, ^bb56(%17 : i64), ^bb59
  ^bb56(%105: i64):  // 2 preds: ^bb55, ^bb57
    %106 = llvm.icmp "slt" %105, %6 : i64
    llvm.cond_br %106, ^bb57, ^bb58
  ^bb57:  // pred: ^bb56
    %107 = llvm.add %105, %15 : i64
    llvm.br ^bb56(%107 : i64)
  ^bb58:  // pred: ^bb56
    %108 = llvm.add %103, %15 : i64
    llvm.br ^bb55(%108 : i64)
  ^bb59:  // pred: ^bb55
    %109 = llvm.add %101, %15 : i64
    llvm.br ^bb54(%109 : i64)
  ^bb60:  // pred: ^bb54
    llvm.call @printFloat(%5) : (f64) -> ()
    llvm.call @printFloat(%4) : (f64) -> ()
    llvm.br ^bb61(%17 : i64)
  ^bb61(%110: i64):  // 2 preds: ^bb60, ^bb64
    %111 = llvm.icmp "slt" %110, %3 : i64
    llvm.cond_br %111, ^bb62(%17 : i64), ^bb65(%17 : i64)
  ^bb62(%112: i64):  // 2 preds: ^bb61, ^bb63
    %113 = llvm.icmp "slt" %112, %3 : i64
    llvm.cond_br %113, ^bb63, ^bb64
  ^bb63:  // pred: ^bb62
    %114 = llvm.call @generate_random() : () -> f64
    %115 = llvm.add %112, %15 : i64
    llvm.br ^bb62(%115 : i64)
  ^bb64:  // pred: ^bb62
    %116 = llvm.add %110, %15 : i64
    llvm.br ^bb61(%116 : i64)
  ^bb65(%117: i64):  // 2 preds: ^bb61, ^bb68
    %118 = llvm.icmp "slt" %117, %3 : i64
    llvm.cond_br %118, ^bb66(%17 : i64), ^bb69(%17 : i64)
  ^bb66(%119: i64):  // 2 preds: ^bb65, ^bb67
    %120 = llvm.icmp "slt" %119, %3 : i64
    llvm.cond_br %120, ^bb67, ^bb68
  ^bb67:  // pred: ^bb66
    %121 = llvm.call @generate_random() : () -> f64
    %122 = llvm.add %119, %15 : i64
    llvm.br ^bb66(%122 : i64)
  ^bb68:  // pred: ^bb66
    %123 = llvm.add %117, %15 : i64
    llvm.br ^bb65(%123 : i64)
  ^bb69(%124: i64):  // 2 preds: ^bb65, ^bb74
    %125 = llvm.icmp "slt" %124, %3 : i64
    llvm.cond_br %125, ^bb70(%17 : i64), ^bb75
  ^bb70(%126: i64):  // 2 preds: ^bb69, ^bb73
    %127 = llvm.icmp "slt" %126, %3 : i64
    llvm.cond_br %127, ^bb71(%17 : i64), ^bb74
  ^bb71(%128: i64):  // 2 preds: ^bb70, ^bb72
    %129 = llvm.icmp "slt" %128, %3 : i64
    llvm.cond_br %129, ^bb72, ^bb73
  ^bb72:  // pred: ^bb71
    %130 = llvm.add %128, %15 : i64
    llvm.br ^bb71(%130 : i64)
  ^bb73:  // pred: ^bb71
    %131 = llvm.add %126, %15 : i64
    llvm.br ^bb70(%131 : i64)
  ^bb74:  // pred: ^bb70
    %132 = llvm.add %124, %15 : i64
    llvm.br ^bb69(%132 : i64)
  ^bb75:  // pred: ^bb69
    llvm.call @printFloat(%2) : (f64) -> ()
    llvm.return %1 : i32
  }
}

