add_library(BoasLib
  frontend/Lexer.cpp
  parser/Parser.cpp
  mlir/MLIRGen.cpp
  analyzer/SemanticAnalyzer.cpp
)

target_include_directories(BoasLib PUBLIC
  ${CMAKE_SOURCE_DIR}/include
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
)

target_link_libraries(BoasLib PRIVATE ${LIBS})

add_executable(boas main.cpp)
target_link_libraries(boas PRIVATE BoasLib ${LIBS})
