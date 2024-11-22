module {
  func.func private @printFloat(f64)
  func.func private @printString(memref<*xi8>)
  func.func private @system_time_msec() -> f64
  func.func private @generate_random() -> f64
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<2x2xf64>
    %cst = arith.constant 1.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    memref.store %cst, %alloc[%c0, %c0_0] : memref<2x2xf64>
    %cst_1 = arith.constant 2.000000e+00 : f64
    %c0_2 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %cst_1, %alloc[%c0_2, %c1] : memref<2x2xf64>
    %cst_3 = arith.constant 2.000000e+00 : f64
    %c1_4 = arith.constant 1 : index
    %c0_5 = arith.constant 0 : index
    memref.store %cst_3, %alloc[%c1_4, %c0_5] : memref<2x2xf64>
    %cst_6 = arith.constant 3.000000e+00 : f64
    %c1_7 = arith.constant 1 : index
    %c1_8 = arith.constant 1 : index
    memref.store %cst_6, %alloc[%c1_7, %c1_8] : memref<2x2xf64>
    %alloc_9 = memref.alloc() : memref<2x2xf64>
    %cst_10 = arith.constant 5.000000e+00 : f64
    %c0_11 = arith.constant 0 : index
    %c0_12 = arith.constant 0 : index
    memref.store %cst_10, %alloc_9[%c0_11, %c0_12] : memref<2x2xf64>
    %cst_13 = arith.constant 1.000000e+00 : f64
    %c0_14 = arith.constant 0 : index
    %c1_15 = arith.constant 1 : index
    memref.store %cst_13, %alloc_9[%c0_14, %c1_15] : memref<2x2xf64>
    %cst_16 = arith.constant 7.000000e+00 : f64
    %c1_17 = arith.constant 1 : index
    %c0_18 = arith.constant 0 : index
    memref.store %cst_16, %alloc_9[%c1_17, %c0_18] : memref<2x2xf64>
    %cst_19 = arith.constant 8.000000e+00 : f64
    %c1_20 = arith.constant 1 : index
    %c1_21 = arith.constant 1 : index
    memref.store %cst_19, %alloc_9[%c1_20, %c1_21] : memref<2x2xf64>
    %alloc_22 = memref.alloc() : memref<2x2xf64>
    %cst_23 = arith.constant 0.000000e+00 : f64
    %c0_24 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c2_25 = arith.constant 2 : index
    %c2_26 = arith.constant 2 : index
    %c1_27 = arith.constant 1 : index
    scf.for %arg0 = %c0_24 to %c2 step %c1_27 {
      scf.for %arg1 = %c0_24 to %c2_25 step %c1_27 {
        memref.store %cst_23, %alloc_22[%arg0, %arg1] : memref<2x2xf64>
      }
    }
    scf.for %arg0 = %c0_24 to %c2 step %c1_27 {
      scf.for %arg1 = %c0_24 to %c2_25 step %c1_27 {
        scf.for %arg2 = %c0_24 to %c2_26 step %c1_27 {
          %0 = memref.load %alloc_22[%arg0, %arg1] : memref<2x2xf64>
          %1 = memref.load %alloc[%arg0, %arg2] : memref<2x2xf64>
          %2 = memref.load %alloc_9[%arg2, %arg1] : memref<2x2xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = arith.addf %0, %3 : f64
          memref.store %4, %alloc_22[%arg0, %arg1] : memref<2x2xf64>
        }
      }
    }
    %c0_28 = arith.constant 0 : index
    %c2_29 = arith.constant 2 : index
    %c2_30 = arith.constant 2 : index
    %c1_31 = arith.constant 1 : index
    %cst_32 = arith.constant 4.000000e+00 : f64
    call @printFloat(%cst_32) : (f64) -> ()
    scf.for %arg0 = %c0_28 to %c2_29 step %c1_31 {
      scf.for %arg1 = %c0_28 to %c2_30 step %c1_31 {
        %0 = memref.load %alloc_22[%arg0, %arg1] : memref<2x2xf64>
        func.call @printFloat(%0) : (f64) -> ()
      }
    }
    %c0_i32_33 = arith.constant 0 : i32
    return %c0_i32_33 : i32
  }
}
