module {
  "builtin.module"() ({
    "func.func"() ({
      // Import: tensor
      func @main() {
        %A = "tensor.from_elements"() ({
          3, 3
        }) {type = tensor<1x2xf64>}
        %B = "tensor.from_elements"() ({
          3, 3,
          3, 3
        }) {type = tensor<2x2xf64>}
        %C = "linalg.matmul"(%A, %B) : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
        "tensor.print"(%C) : (tensor<1x2xf64>)
      }
    }) {sym_name = "main", function_type = () -> (), sym_visibility = "public"}
  }) {}
}