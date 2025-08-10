#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
"builtin.module"() ({
  "func.func"() <{function_type = (f64) -> (), sym_name = "printFloat", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = (memref<*xf64>) -> (), sym_name = "printMemrefF64", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = () -> f64, sym_name = "system_time_usec", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = () -> f64, sym_name = "generate_random", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = () -> memref<?x?xf64>, sym_name = "main"}> ({
    %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x2xf64>
    %1 = "arith.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    "memref.store"(%1, %0, %2, %3) <{nontemporal = false}> : (f64, memref<2x2xf64>, index, index) -> ()
    %4 = "arith.constant"() <{value = 2.000000e+00 : f64}> : () -> f64
    %5 = "arith.constant"() <{value = 0 : index}> : () -> index
    %6 = "arith.constant"() <{value = 1 : index}> : () -> index
    "memref.store"(%4, %0, %5, %6) <{nontemporal = false}> : (f64, memref<2x2xf64>, index, index) -> ()
    %7 = "arith.constant"() <{value = 3.000000e+00 : f64}> : () -> f64
    %8 = "arith.constant"() <{value = 1 : index}> : () -> index
    %9 = "arith.constant"() <{value = 0 : index}> : () -> index
    "memref.store"(%7, %0, %8, %9) <{nontemporal = false}> : (f64, memref<2x2xf64>, index, index) -> ()
    %10 = "arith.constant"() <{value = 4.000000e+00 : f64}> : () -> f64
    %11 = "arith.constant"() <{value = 1 : index}> : () -> index
    %12 = "arith.constant"() <{value = 1 : index}> : () -> index
    "memref.store"(%10, %0, %11, %12) <{nontemporal = false}> : (f64, memref<2x2xf64>, index, index) -> ()
    %13 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x2xf64>
    %14 = "arith.constant"() <{value = 5.000000e+00 : f64}> : () -> f64
    %15 = "arith.constant"() <{value = 0 : index}> : () -> index
    %16 = "arith.constant"() <{value = 0 : index}> : () -> index
    "memref.store"(%14, %13, %15, %16) <{nontemporal = false}> : (f64, memref<2x2xf64>, index, index) -> ()
    %17 = "arith.constant"() <{value = 6.000000e+00 : f64}> : () -> f64
    %18 = "arith.constant"() <{value = 0 : index}> : () -> index
    %19 = "arith.constant"() <{value = 1 : index}> : () -> index
    "memref.store"(%17, %13, %18, %19) <{nontemporal = false}> : (f64, memref<2x2xf64>, index, index) -> ()
    %20 = "arith.constant"() <{value = 7.000000e+00 : f64}> : () -> f64
    %21 = "arith.constant"() <{value = 1 : index}> : () -> index
    %22 = "arith.constant"() <{value = 0 : index}> : () -> index
    "memref.store"(%20, %13, %21, %22) <{nontemporal = false}> : (f64, memref<2x2xf64>, index, index) -> ()
    %23 = "arith.constant"() <{value = 8.000000e+00 : f64}> : () -> f64
    %24 = "arith.constant"() <{value = 1 : index}> : () -> index
    %25 = "arith.constant"() <{value = 1 : index}> : () -> index
    "memref.store"(%23, %13, %24, %25) <{nontemporal = false}> : (f64, memref<2x2xf64>, index, index) -> ()
    %26 = "arith.constant"() <{value = 2 : index}> : () -> index
    %27 = "arith.constant"() <{value = 2 : index}> : () -> index
    %28 = "arith.constant"() <{value = 2 : index}> : () -> index
    %29 = "arith.constant"() <{value = 2 : index}> : () -> index
    %30 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x2xf64>
    %31 = "memref.alloc"(%26, %29) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    "linalg.matmul"(%0, %13, %31) <{indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg21: f64, %arg22: f64, %arg23: f64):
      %164 = "arith.mulf"(%arg21, %arg22) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      %165 = "arith.addf"(%arg23, %164) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%165) : (f64) -> ()
    }) {boas.backend = "npu_optimized", boas.device = "ascend_npu", boas.strategy = "cann_matmul"} : (memref<2x2xf64>, memref<2x2xf64>, memref<?x?xf64>) -> ()
    %32 = "arith.constant"() <{value = 64 : index}> : () -> index
    %33 = "arith.constant"() <{value = 64 : index}> : () -> index
    %34 = "memref.alloc"(%32, %33) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    %35 = "arith.muli"(%32, %33) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %36 = "arith.constant"() <{value = 32 : index}> : () -> index
    %37 = "arith.minsi"(%36, %35) : (index, index) -> index
    %38 = "memref.alloc"(%37) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf64>
    %39 = "arith.constant"() <{value = 0 : index}> : () -> index
    %40 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%39, %37, %40) ({
    ^bb0(%arg20: index):
      %163 = "func.call"() <{callee = @generate_random}> : () -> f64
      "memref.store"(%163, %38, %arg20) <{nontemporal = false}> : (f64, memref<?xf64>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %41 = "arith.constant"() <{value = 0 : index}> : () -> index
    %42 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%41, %35, %42) ({
    ^bb0(%arg19: index):
      %159 = "arith.divui"(%arg19, %33) : (index, index) -> index
      %160 = "arith.remui"(%arg19, %33) : (index, index) -> index
      %161 = "arith.remui"(%arg19, %37) : (index, index) -> index
      %162 = "memref.load"(%38, %161) <{nontemporal = false}> : (memref<?xf64>, index) -> f64
      "memref.store"(%162, %34, %159, %160) <{nontemporal = false}> : (f64, memref<?x?xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "memref.dealloc"(%38) : (memref<?xf64>) -> ()
    %43 = "arith.constant"() <{value = 64 : index}> : () -> index
    %44 = "arith.constant"() <{value = 64 : index}> : () -> index
    %45 = "memref.alloc"(%43, %44) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    %46 = "arith.muli"(%43, %44) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %47 = "arith.constant"() <{value = 32 : index}> : () -> index
    %48 = "arith.minsi"(%47, %46) : (index, index) -> index
    %49 = "memref.alloc"(%48) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf64>
    %50 = "arith.constant"() <{value = 0 : index}> : () -> index
    %51 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%50, %48, %51) ({
    ^bb0(%arg18: index):
      %158 = "func.call"() <{callee = @generate_random}> : () -> f64
      "memref.store"(%158, %49, %arg18) <{nontemporal = false}> : (f64, memref<?xf64>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %52 = "arith.constant"() <{value = 0 : index}> : () -> index
    %53 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%52, %46, %53) ({
    ^bb0(%arg17: index):
      %154 = "arith.divui"(%arg17, %44) : (index, index) -> index
      %155 = "arith.remui"(%arg17, %44) : (index, index) -> index
      %156 = "arith.remui"(%arg17, %48) : (index, index) -> index
      %157 = "memref.load"(%49, %156) <{nontemporal = false}> : (memref<?xf64>, index) -> f64
      "memref.store"(%157, %45, %154, %155) <{nontemporal = false}> : (f64, memref<?x?xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "memref.dealloc"(%49) : (memref<?xf64>) -> ()
    %54 = "arith.constant"() <{value = 0 : index}> : () -> index
    %55 = "memref.dim"(%34, %54) : (memref<?x?xf64>, index) -> index
    %56 = "arith.constant"() <{value = 1 : index}> : () -> index
    %57 = "memref.dim"(%34, %56) : (memref<?x?xf64>, index) -> index
    %58 = "arith.constant"() <{value = 0 : index}> : () -> index
    %59 = "memref.dim"(%45, %58) : (memref<?x?xf64>, index) -> index
    %60 = "arith.constant"() <{value = 1 : index}> : () -> index
    %61 = "memref.dim"(%45, %60) : (memref<?x?xf64>, index) -> index
    %62 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<?x?xf64>
    %63 = "memref.alloc"(%55, %61) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    "linalg.matmul"(%34, %45, %63) <{indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg14: f64, %arg15: f64, %arg16: f64):
      %152 = "arith.mulf"(%arg14, %arg15) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      %153 = "arith.addf"(%arg16, %152) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%153) : (f64) -> ()
    }) {boas.backend = "npu_optimized", boas.device = "ascend_npu", boas.strategy = "cann_matmul"} : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()
    %64 = "arith.constant"() <{value = 256 : index}> : () -> index
    %65 = "arith.constant"() <{value = 256 : index}> : () -> index
    %66 = "memref.alloc"(%64, %65) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    %67 = "arith.muli"(%64, %65) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %68 = "arith.constant"() <{value = 32 : index}> : () -> index
    %69 = "arith.minsi"(%68, %67) : (index, index) -> index
    %70 = "memref.alloc"(%69) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf64>
    %71 = "arith.constant"() <{value = 0 : index}> : () -> index
    %72 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%71, %69, %72) ({
    ^bb0(%arg13: index):
      %151 = "func.call"() <{callee = @generate_random}> : () -> f64
      "memref.store"(%151, %70, %arg13) <{nontemporal = false}> : (f64, memref<?xf64>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %73 = "arith.constant"() <{value = 0 : index}> : () -> index
    %74 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%73, %67, %74) ({
    ^bb0(%arg12: index):
      %147 = "arith.divui"(%arg12, %65) : (index, index) -> index
      %148 = "arith.remui"(%arg12, %65) : (index, index) -> index
      %149 = "arith.remui"(%arg12, %69) : (index, index) -> index
      %150 = "memref.load"(%70, %149) <{nontemporal = false}> : (memref<?xf64>, index) -> f64
      "memref.store"(%150, %66, %147, %148) <{nontemporal = false}> : (f64, memref<?x?xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "memref.dealloc"(%70) : (memref<?xf64>) -> ()
    %75 = "arith.constant"() <{value = 256 : index}> : () -> index
    %76 = "arith.constant"() <{value = 256 : index}> : () -> index
    %77 = "memref.alloc"(%75, %76) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    %78 = "arith.muli"(%75, %76) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %79 = "arith.constant"() <{value = 32 : index}> : () -> index
    %80 = "arith.minsi"(%79, %78) : (index, index) -> index
    %81 = "memref.alloc"(%80) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf64>
    %82 = "arith.constant"() <{value = 0 : index}> : () -> index
    %83 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%82, %80, %83) ({
    ^bb0(%arg11: index):
      %146 = "func.call"() <{callee = @generate_random}> : () -> f64
      "memref.store"(%146, %81, %arg11) <{nontemporal = false}> : (f64, memref<?xf64>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %84 = "arith.constant"() <{value = 0 : index}> : () -> index
    %85 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%84, %78, %85) ({
    ^bb0(%arg10: index):
      %142 = "arith.divui"(%arg10, %76) : (index, index) -> index
      %143 = "arith.remui"(%arg10, %76) : (index, index) -> index
      %144 = "arith.remui"(%arg10, %80) : (index, index) -> index
      %145 = "memref.load"(%81, %144) <{nontemporal = false}> : (memref<?xf64>, index) -> f64
      "memref.store"(%145, %77, %142, %143) <{nontemporal = false}> : (f64, memref<?x?xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "memref.dealloc"(%81) : (memref<?xf64>) -> ()
    %86 = "arith.constant"() <{value = 0 : index}> : () -> index
    %87 = "memref.dim"(%66, %86) : (memref<?x?xf64>, index) -> index
    %88 = "arith.constant"() <{value = 1 : index}> : () -> index
    %89 = "memref.dim"(%66, %88) : (memref<?x?xf64>, index) -> index
    %90 = "arith.constant"() <{value = 0 : index}> : () -> index
    %91 = "memref.dim"(%77, %90) : (memref<?x?xf64>, index) -> index
    %92 = "arith.constant"() <{value = 1 : index}> : () -> index
    %93 = "memref.dim"(%77, %92) : (memref<?x?xf64>, index) -> index
    %94 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<?x?xf64>
    %95 = "memref.alloc"(%87, %93) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    "linalg.matmul"(%66, %77, %95) <{indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):
      %140 = "arith.mulf"(%arg7, %arg8) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      %141 = "arith.addf"(%arg9, %140) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%141) : (f64) -> ()
    }) {boas.backend = "npu_optimized", boas.device = "ascend_npu", boas.strategy = "cann_matmul"} : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()
    %96 = "arith.constant"() <{value = 512 : index}> : () -> index
    %97 = "arith.constant"() <{value = 512 : index}> : () -> index
    %98 = "memref.alloc"(%96, %97) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    %99 = "arith.muli"(%96, %97) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %100 = "arith.constant"() <{value = 32 : index}> : () -> index
    %101 = "arith.minsi"(%100, %99) : (index, index) -> index
    %102 = "memref.alloc"(%101) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf64>
    %103 = "arith.constant"() <{value = 0 : index}> : () -> index
    %104 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%103, %101, %104) ({
    ^bb0(%arg6: index):
      %139 = "func.call"() <{callee = @generate_random}> : () -> f64
      "memref.store"(%139, %102, %arg6) <{nontemporal = false}> : (f64, memref<?xf64>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %105 = "arith.constant"() <{value = 0 : index}> : () -> index
    %106 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%105, %99, %106) ({
    ^bb0(%arg5: index):
      %135 = "arith.divui"(%arg5, %97) : (index, index) -> index
      %136 = "arith.remui"(%arg5, %97) : (index, index) -> index
      %137 = "arith.remui"(%arg5, %101) : (index, index) -> index
      %138 = "memref.load"(%102, %137) <{nontemporal = false}> : (memref<?xf64>, index) -> f64
      "memref.store"(%138, %98, %135, %136) <{nontemporal = false}> : (f64, memref<?x?xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "memref.dealloc"(%102) : (memref<?xf64>) -> ()
    %107 = "arith.constant"() <{value = 512 : index}> : () -> index
    %108 = "arith.constant"() <{value = 512 : index}> : () -> index
    %109 = "memref.alloc"(%107, %108) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    %110 = "arith.muli"(%107, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %111 = "arith.constant"() <{value = 32 : index}> : () -> index
    %112 = "arith.minsi"(%111, %110) : (index, index) -> index
    %113 = "memref.alloc"(%112) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf64>
    %114 = "arith.constant"() <{value = 0 : index}> : () -> index
    %115 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%114, %112, %115) ({
    ^bb0(%arg4: index):
      %134 = "func.call"() <{callee = @generate_random}> : () -> f64
      "memref.store"(%134, %113, %arg4) <{nontemporal = false}> : (f64, memref<?xf64>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %116 = "arith.constant"() <{value = 0 : index}> : () -> index
    %117 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%116, %110, %117) ({
    ^bb0(%arg3: index):
      %130 = "arith.divui"(%arg3, %108) : (index, index) -> index
      %131 = "arith.remui"(%arg3, %108) : (index, index) -> index
      %132 = "arith.remui"(%arg3, %112) : (index, index) -> index
      %133 = "memref.load"(%113, %132) <{nontemporal = false}> : (memref<?xf64>, index) -> f64
      "memref.store"(%133, %109, %130, %131) <{nontemporal = false}> : (f64, memref<?x?xf64>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "memref.dealloc"(%113) : (memref<?xf64>) -> ()
    %118 = "arith.constant"() <{value = 0 : index}> : () -> index
    %119 = "memref.dim"(%98, %118) : (memref<?x?xf64>, index) -> index
    %120 = "arith.constant"() <{value = 1 : index}> : () -> index
    %121 = "memref.dim"(%98, %120) : (memref<?x?xf64>, index) -> index
    %122 = "arith.constant"() <{value = 0 : index}> : () -> index
    %123 = "memref.dim"(%109, %122) : (memref<?x?xf64>, index) -> index
    %124 = "arith.constant"() <{value = 1 : index}> : () -> index
    %125 = "memref.dim"(%109, %124) : (memref<?x?xf64>, index) -> index
    %126 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<?x?xf64>
    %127 = "memref.alloc"(%119, %125) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf64>
    "linalg.matmul"(%98, %109, %127) <{indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %128 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      %129 = "arith.addf"(%arg2, %128) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%129) : (f64) -> ()
    }) {boas.backend = "npu_optimized", boas.device = "ascend_npu", boas.strategy = "cann_matmul"} : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()
    "func.return"(%127) : (memref<?x?xf64>) -> ()
  }) : () -> ()
}) : () -> ()
