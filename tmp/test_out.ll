; ModuleID = 'boas_module'
source_filename = "boas_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fmt = private unnamed_addr constant [6 x i8] c"%.2f\0A\00", align 1
@fmt.1 = private unnamed_addr constant [6 x i8] c"%.2f\0A\00", align 1
@str = private unnamed_addr constant [5 x i8] c"Boas\00", align 1
@fmt.2 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@bool = private unnamed_addr constant [5 x i8] c"true\00", align 1
@fmt.3 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@fmt.4 = private unnamed_addr constant [6 x i8] c"%.2f\0A\00", align 1

define i32 @main() {
entry:
  %0 = call i32 (ptr, ...) @printf(ptr @fmt, double 1.000000e+01)
  %1 = call i32 (ptr, ...) @printf(ptr @fmt.1, double 2.050000e+01)
  %2 = call i32 (ptr, ...) @printf(ptr @fmt.2, ptr @str)
  %3 = call i32 (ptr, ...) @printf(ptr @fmt.3, ptr @bool)
  %4 = call i32 (ptr, ...) @printf(ptr @fmt.4, double 5.100000e+01)
  ret i32 0
}

declare i32 @printf(ptr, ...)
