; ModuleID = 'boas_module'
source_filename = "boas_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @createList(i64)

declare double @listGetItem(ptr, i64)

declare void @listAppend(ptr, double)

declare double @listPop(ptr)

declare i64 @listSize(ptr)

declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  ret i32 0
}
