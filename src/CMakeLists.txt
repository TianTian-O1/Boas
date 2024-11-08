add_library(BoasLib
  frontend/Lexer.cpp
  parser/Parser.cpp
  mlir/MLIRGen.cpp
  analyzer/SemanticAnalyzer.cpp
  llvm/LLVMGen.cpp
)

target_include_directories(BoasLib PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)

set(LIBS
  MLIRAnalysis
  MLIRArithDialect
  MLIRDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  ${llvm_libs}
)

target_link_libraries(BoasLib PUBLIC ${LIBS} ${LLVM_AVAILABLE_LIBS})

add_executable(boas main.cpp)
target_link_libraries(boas PRIVATE BoasLib ${LIBS} ${LLVM_AVAILABLE_LIBS})
