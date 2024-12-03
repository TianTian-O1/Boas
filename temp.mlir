module attributes {sym_visibility = "private"} {
  memref.global "private" @module_name_main : memref<5xi8> = dense<[109, 97, 105, 110, 0]>
  memref.global "private" @module_name_benchmark : memref<10xi8> = dense<[98, 101, 110, 99, 104, 109, 97, 114, 107, 0]>
  memref.global "private" @module_name_tensor : memref<7xi8> = dense<[116, 101, 110, 115, 111, 114, 0]>
  func.func private @import_module(%arg0: memref<?xi8>) {
    return
  }
  %0 = memref.get_global @module_name_tensor : memref<7xi8>
  %cast = memref.cast %0 : memref<7xi8> to memref<?xi8>
  func.call @import_module(%cast) : (memref<?xi8>) -> ()
  %1 = memref.get_global @module_name_benchmark : memref<10xi8>
  %cast_0 = memref.cast %1 : memref<10xi8> to memref<?xi8>
  func.call @import_module(%cast_0) : (memref<?xi8>) -> ()
  %2 = memref.get_global @module_name_main : memref<5xi8>
  %cast_1 = memref.cast %2 : memref<5xi8> to memref<?xi8>
  func.call @import_module(%cast_1) : (memref<?xi8>) -> ()
}
