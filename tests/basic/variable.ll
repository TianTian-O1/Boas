; ModuleID = 'boas_module'
source_filename = "boas_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@0 = private unnamed_addr constant [6 x i8] c"%.6g\0A\00", align 1
@1 = private unnamed_addr constant [6 x i8] c"%.6g\0A\00", align 1
@2 = private unnamed_addr constant [6 x i8] c"Boas\0A\00", align 1
@3 = private unnamed_addr constant [6 x i8] c"true\0A\00", align 1
@4 = private unnamed_addr constant [6 x i8] c"%.6g\0A\00", align 1

declare ptr @createList(i64)

declare double @listGetItem(ptr, i64)

declare void @listAppend(ptr, double)

declare double @listPop(ptr)

declare i64 @listSize(ptr)

declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  %0 = call i32 (ptr, ...) @printf(ptr @0, double 1.000000e+01)
  %1 = call i32 (ptr, ...) @printf(ptr @1, double 2.050000e+01)
  %2 = call i32 (ptr, ...) @printf(ptr @2)
  %3 = call i32 (ptr, ...) @printf(ptr @3)
  %4 = call i32 (ptr, ...) @printf(ptr @4, double 5.100000e+01)
  ret i32 0
}
