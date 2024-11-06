add_library(BoasLib
  frontend/Lexer.cpp
  parser/Parser.cpp
)

target_include_directories(BoasLib PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

add_executable(boas main.cpp)
target_link_libraries(boas PRIVATE BoasLib)
