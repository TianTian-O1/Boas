enable_testing()
find_package(GTest REQUIRED)

add_executable(boas_tests
  test/basic_tests.cpp
)

target_link_libraries(boas_tests
  PRIVATE
  BoasLib
  GTest::GTest
  GTest::Main
)

add_test(NAME boas_tests COMMAND boas_tests)