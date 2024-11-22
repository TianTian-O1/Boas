; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @printFloat(double)

declare void @printString(i64, ptr)

declare double @system_time_msec()

declare double @generate_random()

define i32 @main() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 4) to i64))
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 2, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 2, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 2, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %10 = getelementptr double, ptr %9, i64 0
  store double 1.000000e+00, ptr %10, align 8
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %12 = getelementptr double, ptr %11, i64 1
  store double 2.000000e+00, ptr %12, align 8
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %14 = getelementptr double, ptr %13, i64 2
  store double 2.000000e+00, ptr %14, align 8
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %16 = getelementptr double, ptr %15, i64 3
  store double 3.000000e+00, ptr %16, align 8
  %17 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 4) to i64))
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %17, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, ptr %17, 1
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 0, 2
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 2, 3, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 2, 3, 1
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 2, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 1, 4, 1
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %26 = getelementptr double, ptr %25, i64 0
  store double 5.000000e+00, ptr %26, align 8
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %28 = getelementptr double, ptr %27, i64 1
  store double 1.000000e+00, ptr %28, align 8
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %30 = getelementptr double, ptr %29, i64 2
  store double 7.000000e+00, ptr %30, align 8
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %32 = getelementptr double, ptr %31, i64 3
  store double 8.000000e+00, ptr %32, align 8
  %33 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 4) to i64))
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %33, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, ptr %33, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 0, 2
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 2, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 2, 3, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 2, 4, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 1, 4, 1
  br label %41

41:                                               ; preds = %53, %0
  %42 = phi i64 [ %54, %53 ], [ 0, %0 ]
  %43 = icmp slt i64 %42, 2
  br i1 %43, label %44, label %55

44:                                               ; preds = %47, %41
  %45 = phi i64 [ %52, %47 ], [ 0, %41 ]
  %46 = icmp slt i64 %45, 2
  br i1 %46, label %47, label %53

47:                                               ; preds = %44
  %48 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1
  %49 = mul i64 %42, 2
  %50 = add i64 %49, %45
  %51 = getelementptr double, ptr %48, i64 %50
  store double 0.000000e+00, ptr %51, align 8
  %52 = add i64 %45, 1
  br label %44

53:                                               ; preds = %44
  %54 = add i64 %42, 1
  br label %41

55:                                               ; preds = %89, %41
  %56 = phi i64 [ %90, %89 ], [ 0, %41 ]
  %57 = icmp slt i64 %56, 2
  br i1 %57, label %58, label %91

58:                                               ; preds = %87, %55
  %59 = phi i64 [ %88, %87 ], [ 0, %55 ]
  %60 = icmp slt i64 %59, 2
  br i1 %60, label %61, label %89

61:                                               ; preds = %64, %58
  %62 = phi i64 [ %86, %64 ], [ 0, %58 ]
  %63 = icmp slt i64 %62, 2
  br i1 %63, label %64, label %87

64:                                               ; preds = %61
  %65 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1
  %66 = mul i64 %56, 2
  %67 = add i64 %66, %59
  %68 = getelementptr double, ptr %65, i64 %67
  %69 = load double, ptr %68, align 8
  %70 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %71 = mul i64 %56, 2
  %72 = add i64 %71, %62
  %73 = getelementptr double, ptr %70, i64 %72
  %74 = load double, ptr %73, align 8
  %75 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %76 = mul i64 %62, 2
  %77 = add i64 %76, %59
  %78 = getelementptr double, ptr %75, i64 %77
  %79 = load double, ptr %78, align 8
  %80 = fmul double %74, %79
  %81 = fadd double %69, %80
  %82 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1
  %83 = mul i64 %56, 2
  %84 = add i64 %83, %59
  %85 = getelementptr double, ptr %82, i64 %84
  store double %81, ptr %85, align 8
  %86 = add i64 %62, 1
  br label %61

87:                                               ; preds = %61
  %88 = add i64 %59, 1
  br label %58

89:                                               ; preds = %58
  %90 = add i64 %56, 1
  br label %55

91:                                               ; preds = %55
  call void @printFloat(double 4.000000e+00)
  br label %92

92:                                               ; preds = %105, %91
  %93 = phi i64 [ %106, %105 ], [ 0, %91 ]
  %94 = icmp slt i64 %93, 2
  br i1 %94, label %95, label %107

95:                                               ; preds = %98, %92
  %96 = phi i64 [ %104, %98 ], [ 0, %92 ]
  %97 = icmp slt i64 %96, 2
  br i1 %97, label %98, label %105

98:                                               ; preds = %95
  %99 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1
  %100 = mul i64 %93, 2
  %101 = add i64 %100, %96
  %102 = getelementptr double, ptr %99, i64 %101
  %103 = load double, ptr %102, align 8
  call void @printFloat(double %103)
  %104 = add i64 %96, 1
  br label %95

105:                                              ; preds = %95
  %106 = add i64 %93, 1
  br label %92

107:                                              ; preds = %92
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
