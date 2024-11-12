; ModuleID = 'matrix_module'
source_filename = "matrix_module"

@format = private unnamed_addr constant [35 x i8] c"Matrix:\0A[%.0f, %.0f]\0A[%.0f, %.0f]\0A\00", align 1

define i32 @main() {
entry:
  %matrix = alloca [4 x float], align 4
  %0 = getelementptr [4 x float], ptr %matrix, i32 0, i32 0
  store float 1.000000e+00, ptr %0, align 4
  %1 = getelementptr [4 x float], ptr %matrix, i32 0, i32 1
  store float 2.000000e+00, ptr %1, align 4
  %2 = getelementptr [4 x float], ptr %matrix, i32 0, i32 2
  store float 3.000000e+00, ptr %2, align 4
  %3 = getelementptr [4 x float], ptr %matrix, i32 0, i32 3
  store float 4.000000e+00, ptr %3, align 4
  %matrix1 = alloca [4 x float], align 4
  %4 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 0
  store float 5.000000e+00, ptr %4, align 4
  %5 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 1
  store float 6.000000e+00, ptr %5, align 4
  %6 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 2
  store float 7.000000e+00, ptr %6, align 4
  %7 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 3
  store float 8.000000e+00, ptr %7, align 4
  %result = alloca [4 x float], align 4
  %8 = getelementptr [4 x float], ptr %matrix, i32 0, i32 0
  %9 = load float, ptr %8, align 4
  %10 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 0
  %11 = load float, ptr %10, align 4
  %12 = fmul float %9, %11
  %13 = fadd float 0.000000e+00, %12
  %14 = getelementptr [4 x float], ptr %matrix, i32 0, i32 1
  %15 = load float, ptr %14, align 4
  %16 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 2
  %17 = load float, ptr %16, align 4
  %18 = fmul float %15, %17
  %19 = fadd float %13, %18
  %20 = getelementptr [4 x float], ptr %result, i32 0, i32 0
  store float %19, ptr %20, align 4
  %21 = getelementptr [4 x float], ptr %matrix, i32 0, i32 0
  %22 = load float, ptr %21, align 4
  %23 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 1
  %24 = load float, ptr %23, align 4
  %25 = fmul float %22, %24
  %26 = fadd float 0.000000e+00, %25
  %27 = getelementptr [4 x float], ptr %matrix, i32 0, i32 1
  %28 = load float, ptr %27, align 4
  %29 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 3
  %30 = load float, ptr %29, align 4
  %31 = fmul float %28, %30
  %32 = fadd float %26, %31
  %33 = getelementptr [4 x float], ptr %result, i32 0, i32 1
  store float %32, ptr %33, align 4
  %34 = getelementptr [4 x float], ptr %matrix, i32 0, i32 2
  %35 = load float, ptr %34, align 4
  %36 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 0
  %37 = load float, ptr %36, align 4
  %38 = fmul float %35, %37
  %39 = fadd float 0.000000e+00, %38
  %40 = getelementptr [4 x float], ptr %matrix, i32 0, i32 3
  %41 = load float, ptr %40, align 4
  %42 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 2
  %43 = load float, ptr %42, align 4
  %44 = fmul float %41, %43
  %45 = fadd float %39, %44
  %46 = getelementptr [4 x float], ptr %result, i32 0, i32 2
  store float %45, ptr %46, align 4
  %47 = getelementptr [4 x float], ptr %matrix, i32 0, i32 2
  %48 = load float, ptr %47, align 4
  %49 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 1
  %50 = load float, ptr %49, align 4
  %51 = fmul float %48, %50
  %52 = fadd float 0.000000e+00, %51
  %53 = getelementptr [4 x float], ptr %matrix, i32 0, i32 3
  %54 = load float, ptr %53, align 4
  %55 = getelementptr [4 x float], ptr %matrix1, i32 0, i32 3
  %56 = load float, ptr %55, align 4
  %57 = fmul float %54, %56
  %58 = fadd float %52, %57
  %59 = getelementptr [4 x float], ptr %result, i32 0, i32 3
  store float %58, ptr %59, align 4
  %60 = getelementptr [4 x float], ptr %result, i32 0, i32 0
  %61 = load float, ptr %60, align 4
  %62 = fpext float %61 to double
  %63 = getelementptr [4 x float], ptr %result, i32 0, i32 1
  %64 = load float, ptr %63, align 4
  %65 = fpext float %64 to double
  %66 = getelementptr [4 x float], ptr %result, i32 0, i32 2
  %67 = load float, ptr %66, align 4
  %68 = fpext float %67 to double
  %69 = getelementptr [4 x float], ptr %result, i32 0, i32 3
  %70 = load float, ptr %69, align 4
  %71 = fpext float %70 to double
  %72 = call i32 (ptr, ...) @printf(ptr @format, double %62, double %65, double %68, double %71)
  ret i32 0
}

declare i32 @printf(ptr, ...)
