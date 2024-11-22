module {
  func.func private @printFloat(f64)
  func.func private @printString(memref<*xi8>)
  func.func private @system_time_msec() -> f64
  func.func private @generate_random() -> f64
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f64
    call @printFloat(%cst) : (f64) -> ()
    %alloc = memref.alloc() : memref<64x64xf64>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c64_0 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c64 step %c1 {
      scf.for %arg1 = %c0 to %c64_0 step %c1 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc[%arg0, %arg1] : memref<64x64xf64>
      }
    }
    %alloc_1 = memref.alloc() : memref<64x64xf64>
    %c0_2 = arith.constant 0 : index
    %c64_3 = arith.constant 64 : index
    %c64_4 = arith.constant 64 : index
    %c1_5 = arith.constant 1 : index
    scf.for %arg0 = %c0_2 to %c64_3 step %c1_5 {
      scf.for %arg1 = %c0_2 to %c64_4 step %c1_5 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_1[%arg0, %arg1] : memref<64x64xf64>
      }
    }
    %alloc_6 = memref.alloc() : memref<4096xf64>
    %c0_7 = arith.constant 0 : index
    %c1_8 = arith.constant 1 : index
    %c64_9 = arith.constant 64 : index
    %c64_10 = arith.constant 64 : index
    %c64_11 = arith.constant 64 : index
    %cst_12 = arith.constant 0.000000e+00 : f64
    scf.for %arg0 = %c0_7 to %c64_9 step %c1_8 {
      scf.for %arg1 = %c0_7 to %c64_11 step %c1_8 {
        %0 = scf.for %arg2 = %c0_7 to %c64_10 step %c1_8 iter_args(%arg3 = %cst_12) -> (f64) {
          %3 = arith.muli %arg0, %c64_10 : index
          %4 = arith.addi %3, %arg2 : index
          %5 = arith.muli %arg2, %c64_11 : index
          %6 = arith.addi %5, %arg1 : index
          %7 = memref.load %alloc[%arg0, %arg2] : memref<64x64xf64>
          %8 = memref.load %alloc_1[%arg2, %arg1] : memref<64x64xf64>
          %9 = arith.mulf %7, %8 : f64
          %10 = arith.addf %arg3, %9 : f64
          scf.yield %10 : f64
        }
        %1 = arith.muli %arg0, %c64_11 : index
        %2 = arith.addi %1, %arg1 : index
        memref.store %0, %alloc_6[%2] : memref<4096xf64>
      }
    }
    %cst_13 = arith.constant 2.000000e+00 : f64
    call @printFloat(%cst_13) : (f64) -> ()
    %cst_14 = arith.constant 3.000000e+00 : f64
    call @printFloat(%cst_14) : (f64) -> ()
    %alloc_15 = memref.alloc() : memref<128x128xf64>
    %c0_16 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c128_17 = arith.constant 128 : index
    %c1_18 = arith.constant 1 : index
    scf.for %arg0 = %c0_16 to %c128 step %c1_18 {
      scf.for %arg1 = %c0_16 to %c128_17 step %c1_18 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_15[%arg0, %arg1] : memref<128x128xf64>
      }
    }
    %alloc_19 = memref.alloc() : memref<128x128xf64>
    %c0_20 = arith.constant 0 : index
    %c128_21 = arith.constant 128 : index
    %c128_22 = arith.constant 128 : index
    %c1_23 = arith.constant 1 : index
    scf.for %arg0 = %c0_20 to %c128_21 step %c1_23 {
      scf.for %arg1 = %c0_20 to %c128_22 step %c1_23 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_19[%arg0, %arg1] : memref<128x128xf64>
      }
    }
    %alloc_24 = memref.alloc() : memref<16384xf64>
    %c0_25 = arith.constant 0 : index
    %c1_26 = arith.constant 1 : index
    %c128_27 = arith.constant 128 : index
    %c128_28 = arith.constant 128 : index
    %c128_29 = arith.constant 128 : index
    %cst_30 = arith.constant 0.000000e+00 : f64
    scf.for %arg0 = %c0_25 to %c128_27 step %c1_26 {
      scf.for %arg1 = %c0_25 to %c128_29 step %c1_26 {
        %0 = scf.for %arg2 = %c0_25 to %c128_28 step %c1_26 iter_args(%arg3 = %cst_30) -> (f64) {
          %3 = arith.muli %arg0, %c128_28 : index
          %4 = arith.addi %3, %arg2 : index
          %5 = arith.muli %arg2, %c128_29 : index
          %6 = arith.addi %5, %arg1 : index
          %7 = memref.load %alloc_15[%arg0, %arg2] : memref<128x128xf64>
          %8 = memref.load %alloc_19[%arg2, %arg1] : memref<128x128xf64>
          %9 = arith.mulf %7, %8 : f64
          %10 = arith.addf %arg3, %9 : f64
          scf.yield %10 : f64
        }
        %1 = arith.muli %arg0, %c128_29 : index
        %2 = arith.addi %1, %arg1 : index
        memref.store %0, %alloc_24[%2] : memref<16384xf64>
      }
    }
    %cst_31 = arith.constant 4.000000e+00 : f64
    call @printFloat(%cst_31) : (f64) -> ()
    %cst_32 = arith.constant 5.000000e+00 : f64
    call @printFloat(%cst_32) : (f64) -> ()
    %alloc_33 = memref.alloc() : memref<256x256xf64>
    %c0_34 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c256_35 = arith.constant 256 : index
    %c1_36 = arith.constant 1 : index
    scf.for %arg0 = %c0_34 to %c256 step %c1_36 {
      scf.for %arg1 = %c0_34 to %c256_35 step %c1_36 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_33[%arg0, %arg1] : memref<256x256xf64>
      }
    }
    %alloc_37 = memref.alloc() : memref<256x256xf64>
    %c0_38 = arith.constant 0 : index
    %c256_39 = arith.constant 256 : index
    %c256_40 = arith.constant 256 : index
    %c1_41 = arith.constant 1 : index
    scf.for %arg0 = %c0_38 to %c256_39 step %c1_41 {
      scf.for %arg1 = %c0_38 to %c256_40 step %c1_41 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_37[%arg0, %arg1] : memref<256x256xf64>
      }
    }
    %alloc_42 = memref.alloc() : memref<65536xf64>
    %c0_43 = arith.constant 0 : index
    %c1_44 = arith.constant 1 : index
    %c256_45 = arith.constant 256 : index
    %c256_46 = arith.constant 256 : index
    %c256_47 = arith.constant 256 : index
    %cst_48 = arith.constant 0.000000e+00 : f64
    scf.for %arg0 = %c0_43 to %c256_45 step %c1_44 {
      scf.for %arg1 = %c0_43 to %c256_47 step %c1_44 {
        %0 = scf.for %arg2 = %c0_43 to %c256_46 step %c1_44 iter_args(%arg3 = %cst_48) -> (f64) {
          %3 = arith.muli %arg0, %c256_46 : index
          %4 = arith.addi %3, %arg2 : index
          %5 = arith.muli %arg2, %c256_47 : index
          %6 = arith.addi %5, %arg1 : index
          %7 = memref.load %alloc_33[%arg0, %arg2] : memref<256x256xf64>
          %8 = memref.load %alloc_37[%arg2, %arg1] : memref<256x256xf64>
          %9 = arith.mulf %7, %8 : f64
          %10 = arith.addf %arg3, %9 : f64
          scf.yield %10 : f64
        }
        %1 = arith.muli %arg0, %c256_47 : index
        %2 = arith.addi %1, %arg1 : index
        memref.store %0, %alloc_42[%2] : memref<65536xf64>
      }
    }
    %cst_49 = arith.constant 6.000000e+00 : f64
    call @printFloat(%cst_49) : (f64) -> ()
    %cst_50 = arith.constant 7.000000e+00 : f64
    call @printFloat(%cst_50) : (f64) -> ()
    %alloc_51 = memref.alloc() : memref<512x512xf64>
    %c0_52 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c512_53 = arith.constant 512 : index
    %c1_54 = arith.constant 1 : index
    scf.for %arg0 = %c0_52 to %c512 step %c1_54 {
      scf.for %arg1 = %c0_52 to %c512_53 step %c1_54 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_51[%arg0, %arg1] : memref<512x512xf64>
      }
    }
    %alloc_55 = memref.alloc() : memref<512x512xf64>
    %c0_56 = arith.constant 0 : index
    %c512_57 = arith.constant 512 : index
    %c512_58 = arith.constant 512 : index
    %c1_59 = arith.constant 1 : index
    scf.for %arg0 = %c0_56 to %c512_57 step %c1_59 {
      scf.for %arg1 = %c0_56 to %c512_58 step %c1_59 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_55[%arg0, %arg1] : memref<512x512xf64>
      }
    }
    %alloc_60 = memref.alloc() : memref<262144xf64>
    %c0_61 = arith.constant 0 : index
    %c1_62 = arith.constant 1 : index
    %c512_63 = arith.constant 512 : index
    %c512_64 = arith.constant 512 : index
    %c512_65 = arith.constant 512 : index
    %cst_66 = arith.constant 0.000000e+00 : f64
    scf.for %arg0 = %c0_61 to %c512_63 step %c1_62 {
      scf.for %arg1 = %c0_61 to %c512_65 step %c1_62 {
        %0 = scf.for %arg2 = %c0_61 to %c512_64 step %c1_62 iter_args(%arg3 = %cst_66) -> (f64) {
          %3 = arith.muli %arg0, %c512_64 : index
          %4 = arith.addi %3, %arg2 : index
          %5 = arith.muli %arg2, %c512_65 : index
          %6 = arith.addi %5, %arg1 : index
          %7 = memref.load %alloc_51[%arg0, %arg2] : memref<512x512xf64>
          %8 = memref.load %alloc_55[%arg2, %arg1] : memref<512x512xf64>
          %9 = arith.mulf %7, %8 : f64
          %10 = arith.addf %arg3, %9 : f64
          scf.yield %10 : f64
        }
        %1 = arith.muli %arg0, %c512_65 : index
        %2 = arith.addi %1, %arg1 : index
        memref.store %0, %alloc_60[%2] : memref<262144xf64>
      }
    }
    %cst_67 = arith.constant 8.000000e+00 : f64
    call @printFloat(%cst_67) : (f64) -> ()
    %cst_68 = arith.constant 9.000000e+00 : f64
    call @printFloat(%cst_68) : (f64) -> ()
    %alloc_69 = memref.alloc() : memref<1024x1024xf64>
    %c0_70 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1024_71 = arith.constant 1024 : index
    %c1_72 = arith.constant 1 : index
    scf.for %arg0 = %c0_70 to %c1024 step %c1_72 {
      scf.for %arg1 = %c0_70 to %c1024_71 step %c1_72 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_69[%arg0, %arg1] : memref<1024x1024xf64>
      }
    }
    %alloc_73 = memref.alloc() : memref<1024x1024xf64>
    %c0_74 = arith.constant 0 : index
    %c1024_75 = arith.constant 1024 : index
    %c1024_76 = arith.constant 1024 : index
    %c1_77 = arith.constant 1 : index
    scf.for %arg0 = %c0_74 to %c1024_75 step %c1_77 {
      scf.for %arg1 = %c0_74 to %c1024_76 step %c1_77 {
        %0 = func.call @generate_random() : () -> f64
        memref.store %0, %alloc_73[%arg0, %arg1] : memref<1024x1024xf64>
      }
    }
    %alloc_78 = memref.alloc() : memref<1048576xf64>
    %c0_79 = arith.constant 0 : index
    %c1_80 = arith.constant 1 : index
    %c1024_81 = arith.constant 1024 : index
    %c1024_82 = arith.constant 1024 : index
    %c1024_83 = arith.constant 1024 : index
    %cst_84 = arith.constant 0.000000e+00 : f64
    scf.for %arg0 = %c0_79 to %c1024_81 step %c1_80 {
      scf.for %arg1 = %c0_79 to %c1024_83 step %c1_80 {
        %0 = scf.for %arg2 = %c0_79 to %c1024_82 step %c1_80 iter_args(%arg3 = %cst_84) -> (f64) {
          %3 = arith.muli %arg0, %c1024_82 : index
          %4 = arith.addi %3, %arg2 : index
          %5 = arith.muli %arg2, %c1024_83 : index
          %6 = arith.addi %5, %arg1 : index
          %7 = memref.load %alloc_69[%arg0, %arg2] : memref<1024x1024xf64>
          %8 = memref.load %alloc_73[%arg2, %arg1] : memref<1024x1024xf64>
          %9 = arith.mulf %7, %8 : f64
          %10 = arith.addf %arg3, %9 : f64
          scf.yield %10 : f64
        }
        %1 = arith.muli %arg0, %c1024_83 : index
        %2 = arith.addi %1, %arg1 : index
        memref.store %0, %alloc_78[%2] : memref<1048576xf64>
      }
    }
    %cst_85 = arith.constant 1.000000e+01 : f64
    call @printFloat(%cst_85) : (f64) -> ()
    %c0_i32_86 = arith.constant 0 : i32
    return %c0_i32_86 : i32
  }
}
