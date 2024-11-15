; ModuleID = 'matrix_module'
source_filename = "matrix_module"

@matrixA = private constant [4 x float] [float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00]
@matrixB = private constant [4 x float] [float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00]
@format = private unnamed_addr constant [40 x i8] c"Result matrix:\0A[%.2f %.2f]\0A[%.2f %.2f]\0A\00", align 1

define i32 @main() {
entry:
  %result = alloca [4 x float], align 4
  %0 = load float, ptr @matrixA, align 4
  %1 = load float, ptr @matrixB, align 4
  %2 = fmul float %0, %1
  %3 = fadd float 0.000000e+00, %2
  %4 = load float, ptr getelementptr ([4 x float], ptr @matrixA, i32 0, i32 1), align 4
  %5 = load float, ptr getelementptr ([4 x float], ptr @matrixB, i32 0, i32 2), align 4
  %6 = fmul float %4, %5
  %7 = fadd float %3, %6
  %8 = getelementptr [4 x float], ptr %result, i32 0, i32 0
  store float %7, ptr %8, align 4
  %9 = load float, ptr @matrixA, align 4
  %10 = load float, ptr getelementptr ([4 x float], ptr @matrixB, i32 0, i32 1), align 4
  %11 = fmul float %9, %10
  %12 = fadd float 0.000000e+00, %11
  %13 = load float, ptr getelementptr ([4 x float], ptr @matrixA, i32 0, i32 1), align 4
  %14 = load float, ptr getelementptr ([4 x float], ptr @matrixB, i32 0, i32 3), align 4
  %15 = fmul float %13, %14
  %16 = fadd float %12, %15
  %17 = getelementptr [4 x float], ptr %result, i32 0, i32 1
  store float %16, ptr %17, align 4
  %18 = load float, ptr getelementptr ([4 x float], ptr @matrixA, i32 0, i32 2), align 4
  %19 = load float, ptr @matrixB, align 4
  %20 = fmul float %18, %19
  %21 = fadd float 0.000000e+00, %20
  %22 = load float, ptr getelementptr ([4 x float], ptr @matrixA, i32 0, i32 3), align 4
  %23 = load float, ptr getelementptr ([4 x float], ptr @matrixB, i32 0, i32 2), align 4
  %24 = fmul float %22, %23
  %25 = fadd float %21, %24
  %26 = getelementptr [4 x float], ptr %result, i32 0, i32 2
  store float %25, ptr %26, align 4
  %27 = load float, ptr getelementptr ([4 x float], ptr @matrixA, i32 0, i32 2), align 4
  %28 = load float, ptr getelementptr ([4 x float], ptr @matrixB, i32 0, i32 1), align 4
  %29 = fmul float %27, %28
  %30 = fadd float 0.000000e+00, %29
  %31 = load float, ptr getelementptr ([4 x float], ptr @matrixA, i32 0, i32 3), align 4
  %32 = load float, ptr getelementptr ([4 x float], ptr @matrixB, i32 0, i32 3), align 4
  %33 = fmul float %31, %32
  %34 = fadd float %30, %33
  %35 = getelementptr [4 x float], ptr %result, i32 0, i32 3
  store float %34, ptr %35, align 4
  %36 = getelementptr [4 x float], ptr %result, i32 0, i32 0
  %37 = load float, ptr %36, align 4
  %38 = fpext float %37 to double
  %39 = getelementptr [4 x float], ptr %result, i32 0, i32 1
  %40 = load float, ptr %39, align 4
  %41 = fpext float %40 to double
  %42 = getelementptr [4 x float], ptr %result, i32 0, i32 2
  %43 = load float, ptr %42, align 4
  %44 = fpext float %43 to double
  %45 = getelementptr [4 x float], ptr %result, i32 0, i32 3
  %46 = load float, ptr %45, align 4
  %47 = fpext float %46 to double
  %48 = call i32 (ptr, ...) @printf(ptr @format, double %38, double %41, double %44, double %47)
  ret i32 0
}

declare i32 @printf(ptr, ...)
