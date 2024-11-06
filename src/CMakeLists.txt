add_library(BoasLib
  frontend/Lexer.cpp
  parser/Parser.cpp
)

target_include_directories(BoasLib PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(BoasLib PRIVATE
  MLIRIR
  MLIRSupport
)

add_executable(boas main.cpp)
target_link_libraries(boas PRIVATE 
  BoasLib
  MLIRIR
  MLIRSupport
)
